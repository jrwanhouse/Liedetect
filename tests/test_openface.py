"""Testes do detector OpenFace (openface_backend.py).

Os testes de protocolo usam um FeatureExtraction falso; o teste real roda quando o OpenFace
esta instalado em openface/ (leva ~15 s).

Rodar: python -m unittest discover -s tests -v
"""
import os
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from auxiliares import FELIZ_OF, NEUTRO_OF, linha_openface
import facs
import openface_backend as of
from analise import executar_analise

FPS = 30
FALSO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "falso_openface.py")


def analisar(*partes):
    """AnalisadorOpenFace sobre 1 s neutro de calibracao seguido de (linha, n_frames)."""
    linhas, i = {}, 0
    for linha, n in ((NEUTRO_OF, 30),) + partes:
        for _ in range(n):
            linhas[i] = linha
            i += 1
    a = of.AnalisadorOpenFace(FPS, False, 1.0, linhas)
    frame = np.zeros((480, 640, 3), np.uint8)
    resultados = [a.processar(frame.copy(), j) for j in range(i)]
    a.finalizar()
    return a, resultados


class AUs(unittest.TestCase):
    def test_intensidade_0_a_5_vira_letra_do_facs(self):
        for r, letra in [(1.0, "A"), (2.0, "B"), (3.0, "C"), (4.0, "D"), (5.0, "E")]:
            with self.subTest(intensidade=r):
                _, res = analisar((linha_openface({12: r}), 2))
                self.assertEqual(res[-1].aus[12].intensidade, letra)
        _, res = analisar((linha_openface({12: 0.9}), 2))
        self.assertNotIn(12, res[-1].aus)  # abaixo de 1: ausente

    def test_emocoes_pelas_regras_emfacs(self):
        a, res = analisar((FELIZ_OF, 10))
        self.assertEqual(res[-1].emocoes, {"Felicidade": {6, 12}})
        self.assertEqual(a.episodios[0]["aus"], "AU6C+AU12D")
        self.assertEqual(res[-1].malha, facs.MALHA_68)

    def test_au28_por_presenca(self):
        _, res = analisar((linha_openface(au28=1.0), 2))
        self.assertIn(28, res[-1].aus)

    def test_confianca_baixa_e_sem_rosto(self):
        _, res = analisar((linha_openface({12: 4}, confianca=0.3), 2))
        self.assertIsNone(res[-1].landmarks)

    def test_piscada_pela_abertura_das_palpebras(self):
        _, res = analisar((linha_openface(abertura=(0.1, 0.1)), 3))
        self.assertIn(45, res[-1].aus)

    def test_cabeca_pela_malha_68(self):
        _, res = analisar((linha_openface(nariz=(40, 0)), 2))
        self.assertIn(51, res[-1].aus)


class Extracao(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = os.path.join(cls.tmp.name, "v.mp4")
        out = cv2.VideoWriter(cls.video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
        for i in range(60):
            out.write(np.full((48, 64, 3), i, np.uint8))
        out.release()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_progresso_csv_e_frames_a_partir_de_zero(self):
        saida = os.path.join(self.tmp.name, "a.csv")
        progresso = []
        of.extrair(self.video, saida, FALSO, ao_progresso=lambda p, t: progresso.append(p))
        self.assertEqual(progresso, list(range(0, 101, 10)))
        linhas = of.carregar(saida)
        self.assertEqual(min(linhas), 0)
        self.assertNotIn(45, linhas)  # success 0 fica de fora
        self.assertEqual(len(linhas), 59)
        self.assertAlmostEqual(linhas[40]["AU12_r"], 4.2)
        self.assertEqual([d for d in os.listdir(self.tmp.name) if d.startswith("openface_")], [])

    def test_video_espelhado_passa_copia_desvirada(self):
        entrada = os.path.join(self.tmp.name, "entrada.txt")
        with mock.patch.dict(os.environ, {"FALSO_ENTRADA": entrada}):
            of.extrair(self.video, os.path.join(self.tmp.name, "b.csv"), FALSO, espelhar=True)
        with open(entrada) as f:
            self.assertTrue(f.read().endswith("desespelhado.mp4"))
        self.assertEqual([d for d in os.listdir(self.tmp.name) if d.startswith("openface_")], [])

    def test_erro_do_openface_vira_excecao(self):
        with mock.patch.dict(os.environ, {"FALSO_MODO": "erro"}):
            with self.assertRaisesRegex(RuntimeError, "Failed to open"):
                of.extrair(self.video, os.path.join(self.tmp.name, "c.csv"), FALSO)

    def test_interromper(self):
        saida = os.path.join(self.tmp.name, "d.csv")
        chamadas = iter([True, True])
        with mock.patch.dict(os.environ, {"FALSO_MODO": "lento"}):
            with self.assertRaises(of.AnaliseInterrompida):
                of.extrair(self.video, saida, FALSO, continuar=lambda: next(chamadas, False))
        time.sleep(2)
        self.assertFalse(os.path.exists(saida))
        self.assertEqual([d for d in os.listdir(self.tmp.name) if d.startswith("openface_")], [])

    def test_analise_completa_e_reaproveitamento(self):
        pasta = os.path.join(self.tmp.name, "saida")
        cfg = {"webcam_espelhada": False, "calibracao_segundos": 1.0, "salvar_recortes": False,
               "detector": facs.OPENFACE, "openface_exe": FALSO}
        etapas = []
        dados = executar_analise(self.video, cfg, pasta, ao_etapa=etapas.append)
        self.assertEqual(dados["detector"], facs.OPENFACE)
        self.assertEqual([e["emocao"] for e in dados["episodios"]], ["Felicidade"])
        self.assertIn("OpenFace: 100%", etapas)
        with mock.patch.object(of, "extrair", side_effect=AssertionError("deveria reaproveitar")):
            executar_analise(self.video, cfg, pasta, ao_etapa=etapas.append)
        self.assertTrue(any("reaproveitando" in e for e in etapas))

    def test_sem_openface_avisa(self):
        cfg = {"webcam_espelhada": False, "calibracao_segundos": 1.0, "salvar_recortes": False,
               "detector": facs.OPENFACE, "openface_exe": ""}
        with mock.patch.object(of, "executavel_padrao", return_value=os.path.join(self.tmp.name, "nao_existe.exe")):
            with self.assertRaises(of.OpenFaceIndisponivel):
                executar_analise(self.video, cfg, os.path.join(self.tmp.name, "x"))


@unittest.skipUnless(of.localizar_executavel(), "OpenFace nao instalado em openface/ (veja o README)")
class OpenFaceReal(unittest.TestCase):
    def test_sorriso_vira_au6_au12_e_felicidade(self):
        retrato = os.path.join(os.environ.get("TEMP", tempfile.gettempdir()), "lietest", "portrait.jpg")
        if not os.path.exists(retrato):
            self.skipTest("foto de teste ausente")
        img = cv2.imread(retrato)
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 2 * 2, h // 2 * 2))
        with tempfile.TemporaryDirectory() as tmp:
            video = os.path.join(tmp, "sorriso.mp4")
            out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (img.shape[1], img.shape[0]))
            for _ in range(45):
                out.write(img)
            out.release()
            saida = os.path.join(tmp, "of.csv")
            of.extrair(video, saida, of.localizar_executavel())
            linhas = of.carregar(saida)
            self.assertEqual(len(linhas), 45)
            self.assertGreaterEqual(linhas[40]["AU12_r"], 1.0)
            a = of.AnalisadorOpenFace(FPS, False, 1.0, linhas)
            frame = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
            res = [a.processar(frame.copy(), i) for i in range(45)][-1]
        self.assertTrue({6, 12} <= set(res.aus), res.aus)
        self.assertIn("Felicidade", res.emocoes)


if __name__ == "__main__":
    unittest.main()
