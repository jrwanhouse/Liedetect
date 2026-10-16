"""Testes do detector Py-Feat (pyfeat_backend.py) sem o Py-Feat instalado.

Usa linhas sinteticas do CSV e um worker falso com o mesmo protocolo. O teste com o Py-Feat
de verdade so roda com LIEDETECT_TESTE_PYFEAT=1 (leva minutos em CPU).

Rodar: python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from auxiliares import FELIZ_PF, NEUTRO_PF, linha_pyfeat
import facs
import pyfeat_backend as pf
from analise import executar_analise

FPS = 30
FALSO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "falso_pyfeat_worker.py")


def analisar(linhas, **opcoes):
    """Roda o AnalisadorPyFeat sobre {frame: linha}; retorna (analisador, resultados)."""
    n = max(linhas) + 1
    a = pf.AnalisadorPyFeat(FPS, False, 1.0, linhas, **opcoes)
    frame = np.zeros((480, 640, 3), np.uint8)
    resultados = [a.processar(frame.copy(), i) for i in range(n)]
    a.finalizar()
    return a, resultados


def sequencia(*partes):
    """Sequencia de (linha, n_frames) precedida de 1 s neutro de calibracao."""
    linhas, i = {}, 0
    for linha, n in ((NEUTRO_PF, 30),) + partes:
        for _ in range(n):
            linhas[i] = linha
            i += 1
    return linhas


class AUsEEmocoes(unittest.TestCase):
    def test_probabilidades_viram_aus_com_intensidade(self):
        _, r = analisar(sequencia((FELIZ_PF, 3)))
        aus = r[-1].aus
        self.assertEqual(set(aus), {6, 12})
        self.assertEqual((aus[6].intensidade, aus[12].intensidade), ("D", "E"))
        self.assertEqual(r[-1].malha, facs.MALHA_68)

    def test_emocao_pelo_classificador_do_pyfeat(self):
        _, r = analisar(sequencia((FELIZ_PF, 3)))
        self.assertEqual(r[-1].emocoes, {"Felicidade": {6, 12}})

    def test_emocao_pelas_regras_emfacs(self):
        _, r = analisar(sequencia((FELIZ_PF, 3)), modo_emocoes="emfacs")
        self.assertEqual(r[-1].emocoes, {"Felicidade": {6, 12}})

    def test_classificador_sem_aus_que_expliquem(self):
        so_modelo = linha_pyfeat(emocoes={"fear": 0.9, "neutral": 0.05})
        a, r = analisar(sequencia((so_modelo, 10)))
        self.assertEqual(r[-1].emocoes, {"Medo": set()})
        self.assertEqual(a.episodios[0]["aus"], facs.ROTULO_SEM_AUS)

    def test_neutro_dominante_nao_e_emocao(self):
        _, r = analisar(sequencia((linha_pyfeat(emocoes={"anger": 0.4, "neutral": 0.55}), 3)))
        self.assertEqual(r[-1].emocoes, {})

    def test_au23_e_au24_separadas_no_rotulo(self):
        _, r = analisar(sequencia((linha_pyfeat({23: 0.9, 24: 0.8}), 2)))
        rotulos = {n: facs.rotulo_au(n, au, r[-1].malha) for n, au in r[-1].aus.items()}
        self.assertEqual(rotulos, {23: "AU23E", 24: "AU24D"})

    def test_calibracao_nao_tem_aus(self):
        _, r = analisar(sequencia())
        self.assertTrue(all(x.calibrando and not x.aus for x in r))

    def test_sem_rosto(self):
        _, r = analisar(sequencia((linha_pyfeat(rosto=0.1), 2)))
        self.assertIsNone(r[-1].landmarks)


class Olhos(unittest.TestCase):
    def test_au43_do_modelo_com_olhos_abertos_e_ignorada(self):
        # o AU43 do Py-Feat deu 0,81 num sorriso de olhos abertos (foto real)
        _, r = analisar(sequencia((linha_pyfeat({43: 0.95}), 3)))
        self.assertEqual(set(r[-1].aus) & {43, 45, 46}, set())

    def test_piscada_pela_abertura_das_palpebras(self):
        a, r = analisar(sequencia((linha_pyfeat(abertura=(0.1, 0.1)), 3), (NEUTRO_PF, 5)))
        self.assertIn(45, r[32].aus)
        self.assertEqual(a.atencao.tempos_piscadas, [30 / FPS])

    def test_olhos_fechados_por_mais_de_meio_segundo(self):
        _, r = analisar(sequencia((linha_pyfeat(abertura=(0.1, 0.1)), 20)))
        self.assertIn(43, r[-1].aus)

    def test_piscadela_tem_lado(self):
        _, r = analisar(sequencia((linha_pyfeat(abertura=(0.1, 1.0)), 2)))
        self.assertEqual(r[-1].aus[46].lado, "L")

    def test_olho_fechado_suprime_au5_e_au7(self):
        _, r = analisar(sequencia((linha_pyfeat({5: 0.9, 7: 0.9}, abertura=(0.1, 0.1)), 2)))
        self.assertEqual(set(r[-1].aus) & {5, 7}, set())


class CabecaEPulo(unittest.TestCase):
    def test_giro_da_cabeca_pela_malha_68(self):
        _, r = analisar(sequencia((linha_pyfeat(nariz=(40, 0)), 2)))
        self.assertIn(51, r[-1].aus)
        _, r = analisar(sequencia((linha_pyfeat(nariz=(-40, 0)), 2)))
        self.assertIn(52, r[-1].aus)

    def test_pular_frames_repete_a_ultima_linha(self):
        linhas = {i: l for i, l in sequencia((FELIZ_PF, 9)).items() if i % 3 == 0}
        _, r = analisar(linhas | {38: NEUTRO_PF}, pular=3)
        self.assertEqual([12 in x.aus for x in r[30:36]], [True] * 6)
        self.assertNotIn(12, r[38].aus)


class Extracao(unittest.TestCase):
    """Protocolo com o worker (processo separado), com um worker falso."""

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

    def setUp(self):
        worker = mock.patch.object(pf, "WORKER", FALSO)
        worker.start()
        self.addCleanup(worker.stop)

    def test_extrair_reporta_progresso_e_grava_csv(self):
        saida = os.path.join(self.tmp.name, "a.csv")
        progresso = []
        pf.extrair(self.video, saida, sys.executable, ao_progresso=lambda f, t: progresso.append((f, t)))
        self.assertEqual(progresso[0], (0, 60))
        self.assertEqual(progresso[-1], (60, 60))
        linhas = pf.carregar(saida)
        self.assertEqual(len(linhas), 60)
        self.assertAlmostEqual(linhas[45]["AU12"], 0.9)

    def test_erro_do_worker_vira_excecao(self):
        with mock.patch.dict(os.environ, {"FALSO_MODO": "erro"}):
            with self.assertRaisesRegex(RuntimeError, "modelo nao encontrado"):
                pf.extrair(self.video, os.path.join(self.tmp.name, "b.csv"), sys.executable)

    def test_interromper_nao_deixa_csv_pela_metade(self):
        saida = os.path.join(self.tmp.name, "c.csv")
        chamadas = iter([True, True, False])
        with mock.patch.dict(os.environ, {"FALSO_MODO": "lento"}):
            with self.assertRaises(pf.AnaliseInterrompida):
                pf.extrair(self.video, saida, sys.executable, continuar=lambda: next(chamadas, False))
        time.sleep(3)
        self.assertFalse(os.path.exists(saida))

    def test_cancelar_encerra_o_worker_mesmo_em_silencio(self):
        # cancelar durante um trecho sem mensagens (ex.: carregando modelos) tem de parar o worker
        saida = os.path.join(self.tmp.name, "mudo.csv")
        with mock.patch.dict(os.environ, {"FALSO_MODO": "mudo"}):
            with self.assertRaises(pf.AnaliseInterrompida):
                pf.extrair(self.video, saida, sys.executable, continuar=lambda: False)
        time.sleep(4)
        self.assertFalse(os.path.exists(saida), "o worker continuou rodando depois de cancelado")

    def test_analise_completa_com_pyfeat_e_reaproveitamento(self):
        pasta = os.path.join(self.tmp.name, "saida")
        cfg = {"webcam_espelhada": False, "calibracao_segundos": 1.0, "salvar_recortes": False,
               "detector": facs.PYFEAT, "pyfeat_python": sys.executable, "pyfeat_pular": 1}
        etapas = []
        dados = executar_analise(self.video, cfg, pasta, ao_etapa=etapas.append)
        self.assertEqual(dados["detector"], facs.PYFEAT)
        self.assertEqual([e["emocao"] for e in dados["episodios"]], ["Felicidade"])
        self.assertAlmostEqual(dados["episodios"][0]["inicio"], 1.0, delta=1 / FPS)
        self.assertEqual(dados["resultados"][45].malha, facs.MALHA_68)
        self.assertTrue(any("Py-Feat: 60/60" in e for e in etapas), etapas)

        etapas.clear()
        with mock.patch.object(pf, "extrair", side_effect=AssertionError("deveria reaproveitar")):
            executar_analise(self.video, cfg, pasta, ao_etapa=etapas.append)
        self.assertTrue(any("reaproveitando" in e for e in etapas), etapas)

    def test_sem_ambiente_do_pyfeat_avisa(self):
        cfg = {"webcam_espelhada": False, "calibracao_segundos": 1.0, "salvar_recortes": False,
               "detector": facs.PYFEAT, "pyfeat_python": ""}
        with mock.patch.object(pf, "python_padrao", return_value=os.path.join(self.tmp.name, "nao_existe")):
            with self.assertRaises(pf.PyFeatIndisponivel):
                executar_analise(self.video, cfg, os.path.join(self.tmp.name, "x"))


@unittest.skipUnless(os.environ.get("LIEDETECT_TESTE_PYFEAT") == "1" and pf.localizar_python(),
                     "defina LIEDETECT_TESTE_PYFEAT=1 (precisa do .venv-pyfeat; leva minutos)")
class PyFeatReal(unittest.TestCase):
    def test_extrai_aus_e_emocao_de_um_sorriso(self):
        retrato = os.path.join(os.environ.get("TEMP", tempfile.gettempdir()), "lietest", "portrait.jpg")
        if not os.path.exists(retrato):
            self.skipTest("foto de teste ausente")
        with tempfile.TemporaryDirectory() as tmp:
            video = os.path.join(tmp, "sorriso.mp4")
            img = cv2.imread(retrato)
            h, w = img.shape[:2]
            out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w // 2 * 2, h // 2 * 2))
            for _ in range(4):
                out.write(cv2.resize(img, (w // 2 * 2, h // 2 * 2)))
            out.release()
            saida = os.path.join(tmp, "pf.csv")
            pf.extrair(video, saida, pf.localizar_python())
            linhas = pf.carregar(saida)
        self.assertEqual(len(linhas), 4)
        linha = linhas[0]
        self.assertGreater(linha["AU12"], 0.5)
        self.assertGreater(linha["happiness"], 0.5)
        self.assertIn("x_67", linha)


if __name__ == "__main__":
    unittest.main()
