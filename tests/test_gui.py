"""Testes da interface grafica (PySide6) sem janela visivel: fluxo completo de projeto.

Rodar: python -m unittest discover -s tests -v
"""
import os
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import auxiliares  # noqa: E402  (aplica compat antes do PySide6)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

import facs  # noqa: E402
import gui  # noqa: E402
import preferencias  # noqa: E402
import projeto  # noqa: E402
from test_medo import MEDO  # noqa: E402

FPS = 30


def criar_video(caminho, frames):
    out = cv2.VideoWriter(caminho, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    for i in range(frames):
        out.write(np.full((48, 64, 3), i % 255, np.uint8))
    out.release()


def esperar(condicao, limite=30.0):
    app = QApplication.instance()
    fim = time.monotonic() + limite
    while not condicao():
        if time.monotonic() > fim:
            raise TimeoutError("condicao nao atingida")
        app.processEvents()
        time.sleep(0.01)


class DetectorRealComPySide6(unittest.TestCase):
    """O MediaPipe precisa funcionar numa QThread do PySide6 (ver compat.py)."""

    @unittest.skipUnless(os.path.exists(facs.MODELO_PATH), "modelo face_landmarker.task nao baixado")
    def test_cria_detector_real_numa_qthread(self):
        from PySide6.QtCore import QThread
        resultado = {}

        class Tarefa(QThread):
            def run(self):
                try:
                    detector = facs.criar_detector()
                    detector.close()
                    resultado["ok"] = True
                except Exception as e:  # noqa: BLE001
                    resultado["erro"] = repr(e)

        QApplication.instance() or QApplication([])
        tarefa = Tarefa()
        tarefa.start()
        tarefa.wait()
        self.assertEqual(resultado, {"ok": True})


class FluxoDeProjeto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = os.path.join(cls.tmp.name, "entrevista.mp4")
        # 1 s de calibracao, 1 s neutro, 1 s de medo, 1 s neutro
        cls.sequencia = [auxiliares.NEUTRO] * 60 + [MEDO] * 30 + [auxiliares.NEUTRO] * 30
        criar_video(cls.video, len(cls.sequencia))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        detector = mock.patch.object(facs, "criar_detector",
                                     lambda: auxiliares.DetectorFalso(self.sequencia))
        detector.start()
        self.addCleanup(detector.stop)
        prefs = dict(preferencias.PADRAO, modo_player="revisao", volume=0, salvar_recortes=False)
        self.janela = gui.MainWindow(prefs, os.path.join(self.tmp.name, "prefs.json"))
        self.addCleanup(self._fechar)

    def _fechar(self):
        if self.janela.projeto:
            self.janela.projeto.modificado = False
        self.janela.close()

    def novo_projeto(self, nome):
        caminho = os.path.join(self.tmp.name, nome)
        with mock.patch.object(QFileDialog, "getOpenFileName", return_value=(self.video, "")), \
             mock.patch.object(QFileDialog, "getSaveFileName", return_value=(caminho, "")):
            self.janela.novo_projeto()
        return caminho + projeto.EXTENSAO

    def analisar(self):
        self.janela.analisar()
        esperar(lambda: self.janela.projeto.analise is not None and not self.janela._analisando())

    def test_criar_analisar_marcar_salvar_e_reabrir(self):
        caminho = self.novo_projeto("caso")
        self.assertTrue(os.path.exists(caminho))
        self.assertEqual(self.janela.btn_analisar.text(), "Analisar")

        self.analisar()
        proj = self.janela.projeto
        self.assertEqual([e["emocao"] for e in proj.analise["episodios"]], ["Medo"])
        self.assertAlmostEqual(proj.analise["episodios"][0]["inicio"], 2.0, delta=1 / FPS)
        self.assertIsNotNone(self.janela.revisao, "player de revisao deveria abrir")
        self.assertEqual(self.janela.btn_analisar.text(), "Analisar novamente")
        self.assertFalse(proj.modificado, "analise deveria ser salva automaticamente")

        # perguntas no player de revisao recalculam o resumo
        for t, texto in [(0.5, "Nome?"), (1.8, "Onde estava?"), (3.2, "")]:
            self.janela.ir_para(t)
            self.janela.txt_pergunta.setText(texto)
            self.janela.marcar_pergunta()
        self.assertTrue(self.janela.windowTitle().endswith("*"))
        self.assertEqual(self.janela.list_resumo.count(), 3)
        self.assertTrue(self.janela.salvar_projeto())

        reaberto = projeto.abrir(caminho)
        self.assertEqual([texto for _, texto in sorted(reaberto.perguntas)], ["Nome?", "Onde estava?", ""])
        self.assertEqual(len(reaberto.resultados), len(self.sequencia))

        # outra janela abre o projeto ja analisado com tudo restaurado
        segunda = gui.MainWindow(self.janela.prefs, self.janela.caminho_prefs)
        segunda.abrir_projeto(caminho)
        self.assertIsNotNone(segunda.revisao)
        self.assertEqual(segunda.list_resumo.count(), 3)
        textos = [segunda.list_alerts.item(i).text() for i in range(segunda.list_alerts.count())]
        self.assertTrue(any("Medo" in t for t in textos), textos)

        # remover pergunta com a lista de eventos
        item = next(segunda.list_alerts.item(i) for i in range(segunda.list_alerts.count())
                    if "Onde estava" in segunda.list_alerts.item(i).text())
        segunda.list_alerts.setCurrentItem(item)
        segunda.remover_pergunta()
        self.assertEqual(len(segunda.projeto.perguntas), 2)
        segunda.projeto.modificado = False
        segunda.close()

    def test_menu_de_validacao_exporta_e_avalia(self):
        import validacao
        self.novo_projeto("validar")
        self.analisar()
        pasta = os.path.join(self.tmp.name, "amostra_gui")
        with mock.patch.object(gui.QInputDialog, "getInt", return_value=(20, True)), \
             mock.patch.object(QFileDialog, "getExistingDirectory", return_value=pasta), \
             mock.patch.object(QMessageBox, "information") as info:
            self.janela.exportar_validacao()
        info.assert_called_once()
        self.assertTrue(os.path.exists(os.path.join(pasta, validacao.ARQUIVO_MODELO)))

        planilha = os.path.join(pasta, "codificador.csv")
        with open(os.path.join(pasta, validacao.ARQUIVO_MODELO), encoding="utf-8-sig") as f, \
             open(planilha, "w", encoding="utf-8-sig") as g:
            g.write(f.read())
        dialogos = []
        with mock.patch.object(QFileDialog, "getExistingDirectory", return_value=pasta), \
             mock.patch.object(QFileDialog, "getOpenFileNames", return_value=([planilha], "")), \
             mock.patch.object(gui.DialogoValidacao, "exec", lambda d: dialogos.append(d)):
            self.janela.avaliar_validacao()
        self.assertEqual(len(dialogos), 1)
        self.assertTrue(os.path.exists(os.path.join(pasta, validacao.ARQUIVO_RELATORIO)))

    def test_exportar_validacao_sem_analise_avisa(self):
        self.novo_projeto("sem_analise")
        with mock.patch.object(QMessageBox, "information") as info:
            self.janela.exportar_validacao()
        info.assert_called_once()

    def test_projeto_com_pyfeat(self):
        import sys
        import pyfeat_backend
        falso = os.path.join(os.path.dirname(os.path.abspath(__file__)), "falso_pyfeat_worker.py")
        self.janela.prefs.update(detector="pyfeat", pyfeat_python=sys.executable, pyfeat_pular=1)
        with mock.patch.object(pyfeat_backend, "WORKER", falso):
            self.novo_projeto("com_pyfeat")
            self.assertEqual(self.janela.projeto.detector, "pyfeat")
            self.assertIn("Detector: Py-Feat", self.janela.lbl_projeto.text())
            self.analisar()
        analise = self.janela.projeto.analise
        self.assertEqual(analise["detector"], "pyfeat")
        self.assertEqual([e["emocao"] for e in analise["episodios"]], ["Felicidade"])
        self.assertIsNotNone(self.janela.revisao)
        self.janela.ir_para(2.5)  # desenha as AUs na malha de 68 pontos sem erro
        self.assertEqual(self.janela.projeto.resultados[75].malha, facs.MALHA_68)

    def test_fechar_com_alteracoes_pergunta_antes(self):
        self.novo_projeto("alterado")
        self.janela.chk_webcam.setChecked(not self.janela.chk_webcam.isChecked())
        self.assertTrue(self.janela.projeto.modificado)
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel):
            self.assertFalse(self.janela.close())
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Discard):
            self.assertTrue(self.janela.close())

    def test_trocar_de_projeto_durante_a_analise_nao_mistura_resultados(self):
        primeiro = self.novo_projeto("primeiro")
        self.analisar()
        self.novo_projeto("segundo")
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.janela.analisar()
        self.janela.abrir_projeto(primeiro)
        esperar(lambda: not self.janela._analisando(), limite=10)
        for _ in range(20):
            self.app.processEvents()
        self.assertTrue(self.janela.projeto.caminho.endswith("primeiro" + projeto.EXTENSAO))
        self.assertIsNone(projeto.abrir(self.janela.projeto.caminho.replace("primeiro", "segundo")).analise)

    def test_projeto_inexistente_avisa(self):
        with mock.patch.object(QMessageBox, "warning") as aviso:
            self.janela.abrir_projeto(os.path.join(self.tmp.name, "nao_existe.liedetect"))
        aviso.assert_called_once()
        self.assertIsNone(self.janela.projeto)


if __name__ == "__main__":
    unittest.main()
