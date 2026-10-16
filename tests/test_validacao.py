"""Testes da validacao contra codificacao FACS humana (validacao.py).

Rodar: python -m unittest discover -s tests -v
"""
import csv
import json
import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import auxiliares
import facs
import preferencias
import projeto
import validacao
from analise import executar_analise, separar_resultados

FPS = 30
FRACO, FORTE = 0.45, 0.85  # mouthSmile: o sistema (limiar 0.3) marca os dois; os codificadores, so o forte


class Metricas(unittest.TestCase):
    def test_kappa(self):
        self.assertAlmostEqual(validacao.kappa([1, 1, 0, 0], [1, 0, 0, 0]), 0.5)
        self.assertEqual(validacao.kappa([1, 0, 1], [1, 0, 1]), 1.0)
        self.assertIsNone(validacao.kappa([], []))

    def test_metricas(self):
        m = validacao.metricas([True, True, False, False], [True, False, True, False])
        self.assertEqual((m["vp"], m["fp"], m["fn"], m["vn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["f1"], 0.5)

    def test_celulas(self):
        self.assertEqual(validacao.ler_celula(""), (False, None))
        self.assertEqual(validacao.ler_celula("0"), (False, None))
        self.assertEqual(validacao.ler_celula("x"), (True, None))
        self.assertEqual(validacao.ler_celula(" c "), (True, 3))
        with self.assertRaises(ValueError):
            validacao.ler_celula("talvez")


class FluxoDeValidacao(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = cls.tmp.name
        # 1 s de calibracao + 16 ciclos de (neutro, sorriso fraco, sorriso forte), 0,5 s cada
        sorriso = lambda v: auxiliares.NEUTRO | {"mouthSmileLeft": v, "mouthSmileRight": v}
        cls.verdade_por_frame = [False] * 30
        sequencia = [auxiliares.NEUTRO] * 30
        for _ in range(16):
            for nivel in (None, FRACO, FORTE):
                sequencia += [sorriso(nivel) if nivel else auxiliares.NEUTRO] * 15
                cls.verdade_por_frame += [nivel == FORTE] * 15
        video = os.path.join(base, "v.mp4")
        out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
        for i in range(len(sequencia)):
            out.write(np.full((48, 64, 3), i % 255, np.uint8))
        out.release()

        cls.proj = projeto.novo(video, preferencias.PADRAO, os.path.join(base, "caso.liedetect"))
        with mock.patch.object(facs, "criar_detector", lambda: auxiliares.DetectorFalso(sequencia)):
            dados = executar_analise(video, cls.proj.configuracoes(), cls.proj.pasta_saida())
        cls.proj.analise, cls.proj.resultados = separar_resultados(dados)
        cls.pasta = os.path.join(base, "validacao")
        cls.n = validacao.exportar_amostra(cls.proj, cls.pasta, n=40, semente=7)

    @classmethod
    def tearDownClass(cls):
        facs.carregar_limiares(os.path.join(cls.tmp.name, "nao_existe.json"))  # volta aos padroes
        cls.tmp.cleanup()

    def _codificar(self, nome, erros=()):
        """Planilha de um codificador simulado; `erros` = amostras com AU12 trocada."""
        with open(os.path.join(self.pasta, validacao.ARQUIVO_PREDICOES), encoding="utf-8-sig") as f:
            frame_de = {l["amostra"]: int(l["frame"]) for l in csv.DictReader(f, delimiter=";")}
        with open(os.path.join(self.pasta, validacao.ARQUIVO_MODELO), encoding="utf-8-sig") as f:
            leitor = csv.DictReader(f, delimiter=";")
            campos, linhas = leitor.fieldnames, list(leitor)
        for linha in linhas:
            presente = self.verdade_por_frame[frame_de[linha["amostra"]]] != (linha["amostra"] in erros)
            linha["AU12"] = "D" if presente else ""
        caminho = os.path.join(self.pasta, nome)
        with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, campos, delimiter=";")
            wr.writeheader()
            wr.writerows(linhas)
        return caminho

    def test_exportacao_e_cega(self):
        self.assertGreaterEqual(self.n, 30)
        arquivos = os.listdir(os.path.join(self.pasta, validacao.PASTA_AMOSTRAS))
        self.assertEqual(sum(a.endswith(".png") for a in arquivos), self.n)
        self.assertEqual(sum(a.endswith(".mp4") for a in arquivos), self.n)
        with open(os.path.join(self.pasta, validacao.ARQUIVO_MODELO), encoding="utf-8-sig") as f:
            modelo = f.read()
        self.assertNotIn("score", modelo)              # a planilha dos codificadores nao revela o sistema
        with open(os.path.join(self.pasta, validacao.ARQUIVO_MODELO), encoding="utf-8-sig") as f:
            self.assertTrue(all(l["AU12"] == "" for l in csv.DictReader(f, delimiter=";")))
        frames = [int(l["frame"]) for l in csv.DictReader(
            open(os.path.join(self.pasta, validacao.ARQUIVO_PREDICOES), encoding="utf-8-sig"), delimiter=";")]
        self.assertTrue(all(b - a >= FPS * validacao.ESPACO_MINIMO_S for a, b in zip(frames, frames[1:])))
        self.assertTrue(all(f >= 30 for f in frames), "frames da calibracao nao entram na amostra")

    def test_avaliacao_detecta_falsos_positivos_e_sugere_limiar(self):
        planilhas = [self._codificar("cod1.csv"), self._codificar("cod2.csv"), self._codificar("cod3.csv", erros={"a005"})]
        resultado = {r["au"]: r for r in validacao.avaliar(self.pasta, planilhas)}
        au12 = resultado[12]
        self.assertGreaterEqual(au12["positivos"], validacao.MIN_POSITIVOS)
        self.assertEqual(au12["fn"], 0)
        self.assertGreater(au12["fp"], 0, "sorrisos fracos deveriam ser falsos positivos no limiar 0.3")
        self.assertLess(au12["f1"], 1.0)
        self.assertTrue(FRACO <= au12["limiar_sugerido"] < FORTE, au12["limiar_sugerido"])
        self.assertEqual(au12["f1_sugerido"], 1.0)
        self.assertTrue(au12["sugerir"])
        self.assertTrue(au12["situacao"].startswith("validada"), au12["situacao"])
        self.assertLess(au12["kappa_codificadores"], 1.0)   # o terceiro codificador discorda numa amostra
        self.assertEqual(au12["erro_intensidade"], 0.0)      # forte = D no sistema e nos codificadores
        self.assertTrue(resultado[4]["situacao"].startswith("amostra insuficiente"))
        self.assertTrue(os.path.exists(os.path.join(self.pasta, validacao.ARQUIVO_RELATORIO)))

        destino = os.path.join(self.tmp.name, "limiares_aus.json")
        ajustes = validacao.aplicar_limiares_sugeridos(self.pasta, destino)
        self.assertEqual(ajustes, {12: au12["limiar_sugerido"]})
        self.assertEqual(facs.limiar_de(12), au12["limiar_sugerido"])
        with open(destino, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"12": au12["limiar_sugerido"]})

    def test_empate_entre_codificadores_fica_fora(self):
        planilhas = [self._codificar("e1.csv"), self._codificar("e2.csv", erros={"a001", "a002"})]
        au12 = {r["au"]: r for r in validacao.avaliar(self.pasta, planilhas)}[12]
        self.assertEqual(au12["empates"], 2)
        self.assertEqual(au12["amostras"], self.n - 2)


class ValidacaoComPyFeat(unittest.TestCase):
    """Projetos Py-Feat validam as AUs que o Py-Feat mede e gravam limiares proprios."""

    def test_limiares_vao_para_o_arquivo_do_pyfeat(self):
        import sys
        import pyfeat_backend as pf
        falso = os.path.join(os.path.dirname(os.path.abspath(__file__)), "falso_pyfeat_worker.py")
        with tempfile.TemporaryDirectory() as base:
            video = os.path.join(base, "v.mp4")
            out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
            for i in range(240):
                out.write(np.full((48, 64, 3), i % 255, np.uint8))
            out.release()
            prefs = dict(preferencias.PADRAO, detector="pyfeat")
            proj = projeto.novo(video, prefs, os.path.join(base, "pf.liedetect"))
            cfg = proj.configuracoes() | {"pyfeat_python": sys.executable, "salvar_recortes": False}
            with mock.patch.object(pf, "WORKER", falso):
                dados = executar_analise(video, cfg, proj.pasta_saida())
            proj.analise, proj.resultados = separar_resultados(dados)
            pasta = os.path.join(base, "val")
            validacao.exportar_amostra(proj, pasta, n=12)
            with open(os.path.join(pasta, validacao.ARQUIVO_META), encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["detector"], "pyfeat")
            self.assertIn(24, meta["aus"])                     # o Py-Feat mede a AU24
            self.assertEqual(meta["limiares"]["12"], facs.LIMIAR_PYFEAT)

            with open(os.path.join(pasta, validacao.ARQUIVO_LIMIARES_SUGERIDOS), "w") as f:
                json.dump({"12": 0.7}, f)
            destino = os.path.join(base, "limiares_aus_pyfeat.json")
            try:
                self.assertEqual(validacao.aplicar_limiares_sugeridos(pasta, destino), {12: 0.7})
                self.assertEqual(facs.limiar_de(12, facs.PYFEAT), 0.7)
                self.assertEqual(facs.limiar_de(12, facs.MEDIAPIPE), facs.LIMIAR_PADRAO)  # MediaPipe intocado
            finally:
                facs.carregar_limiares(os.path.join(base, "nao_existe.json"), facs.PYFEAT)

    def test_mediapipe_nao_exporta_aus_que_nao_mede(self):
        self.assertNotIn(24, [au for au in validacao.AUS_FACIAIS if au in facs.AUS_MEDIDAS[facs.MEDIAPIPE]])


if __name__ == "__main__":
    unittest.main()
