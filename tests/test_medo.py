"""Testes da deteccao de Medo: regras EMFACS, blendshapes -> AUs -> emocao e sequencia de video.

Rodar: python -m unittest discover -s tests -v
"""
import unittest

from auxiliares import NEUTRO, analisar_sequencia
from facs import AU, classificar_emocoes

# Prototipo de medo (Ekman): AU1+2+4+5+7+20+26
MEDO = NEUTRO | {
    "browInnerUp": 0.85, "browOuterUpLeft": 0.75, "browOuterUpRight": 0.72,
    "browDownLeft": 0.75, "browDownRight": 0.73,      # AU4: sobrancelhas puxadas juntas
    "eyeWideLeft": 0.80, "eyeWideRight": 0.78,        # AU5
    "eyeSquintLeft": 0.60, "eyeSquintRight": 0.58,    # AU7
    "mouthStretchLeft": 0.70, "mouthStretchRight": 0.68,  # AU20
    "jawOpen": 0.45,                                   # AU26
}


def aus(*nums, lados=None):
    lados = lados or {}
    return {n: AU(0.6, "C", lados.get(n, "")) for n in nums}


class RegrasMedo(unittest.TestCase):
    """Combinacoes de AUs -> emocao (tabela EMOCOES)."""

    def assertSoMedo(self, *nums):
        resultado = classificar_emocoes(aus(*nums))
        self.assertEqual(set(resultado), {"Medo"}, f"AUs {nums} -> {resultado}")

    def test_prototipo_completo(self):
        self.assertSoMedo(1, 2, 4, 5, 7, 20, 26)

    def test_variantes_ekman(self):
        for combo in [(1, 2, 4, 5, 20), (1, 2, 4, 5), (1, 2, 4, 20), (5, 20), (1, 2, 4, 5, 20, 26)]:
            with self.subTest(combo=combo):
                self.assertSoMedo(*combo)

    def test_medo_com_boca_aberta_prevalece_sobre_surpresa(self):
        # 1+2+4+5 com 25/26/27 e medo (Ekman & Friesen); a AU4 distingue da surpresa
        for boca in (25, 26, 27):
            with self.subTest(boca=boca):
                self.assertSoMedo(1, 2, 4, 5, boca)

    def test_sem_au4_e_surpresa(self):
        self.assertEqual(set(classificar_emocoes(aus(1, 2, 5, 26))), {"Surpresa"})

    def test_combinacoes_incompletas_nao_sao_medo(self):
        for combo in [(1, 2), (1, 4), (4,), (5,), (20,), (1, 2, 4), (4, 5, 7)]:
            with self.subTest(combo=combo):
                self.assertNotIn("Medo", classificar_emocoes(aus(*combo)))


class BlendshapesParaMedo(unittest.TestCase):
    """Blendshapes do modelo -> calibracao -> AUs -> Medo."""

    def test_rosto_de_medo_vira_aus_e_medo(self):
        _, resultados = analisar_sequencia([NEUTRO] * 30 + [MEDO] * 5)
        r = resultados[-1]
        self.assertTrue({1, 2, 4, 5, 7, 20, 26} <= set(r.aus), f"AUs detectadas: {sorted(r.aus)}")
        self.assertEqual(set(r.emocoes), {"Medo"})                # nem Surpresa nem Raiva
        self.assertEqual(r.emocoes["Medo"], {1, 2, 4, 5, 7, 20, 26})  # prototipo completo

    def test_calibracao_remove_vies_do_modelo(self):
        # browDown 0.35 no neutro nao pode virar AU4 depois da calibracao
        _, resultados = analisar_sequencia([NEUTRO] * 35)
        self.assertTrue(all(r.calibrando for r in resultados[:30]))
        self.assertEqual(resultados[-1].aus, {})
        self.assertEqual(resultados[-1].emocoes, {})

    def test_intensidade_acompanha_a_expressao(self):
        fraco = {k: NEUTRO[k] + (v - NEUTRO[k]) * 0.6 for k, v in MEDO.items()}
        _, forte = analisar_sequencia([NEUTRO] * 30 + [MEDO] * 3)
        _, leve = analisar_sequencia([NEUTRO] * 30 + [fraco] * 3)
        self.assertIn("Medo", leve[-1].emocoes)
        self.assertGreater(forte[-1].aus[1].intensidade, leve[-1].aus[1].intensidade)

    def test_video_espelhado_nao_altera_medo(self):
        _, resultados = analisar_sequencia([NEUTRO] * 30 + [MEDO] * 5, espelhado=True)
        self.assertIn("Medo", resultados[-1].emocoes)


class SequenciaDeVideo(unittest.TestCase):
    """Episodios de medo ao longo do tempo, incluindo microexpressao."""

    def test_medo_rapido_e_sustentado(self):
        fps = 30
        sequencia = ([NEUTRO] * 30            # 0.0-1.0 s calibracao
                     + [NEUTRO] * 30          # 1.0-2.0 s neutro
                     + [MEDO] * 6             # 2.0-2.2 s medo rapido (200 ms)
                     + [NEUTRO] * 60          # 2.2-4.2 s neutro
                     + [MEDO] * 45            # 4.2-5.7 s medo sustentado (1,5 s)
                     + [NEUTRO] * 30)
        analisador, _ = analisar_sequencia(sequencia, fps)
        medos = [e for e in analisador.episodios if e["emocao"] == "Medo"]
        self.assertEqual(len(medos), 2, analisador.episodios)
        rapido, sustentado = medos
        self.assertAlmostEqual(rapido["inicio"], 2.0, delta=1 / fps)
        self.assertAlmostEqual(sustentado["inicio"], 4.2, delta=1 / fps)
        self.assertAlmostEqual(sustentado["fim"] - sustentado["inicio"], 1.5, delta=0.1)
        self.assertIn("AU4", rapido["aus"])
        self.assertEqual([e for e in analisador.episodios if e["emocao"] == "Surpresa"], [])

        micro = [p for p in analisador.atencao.pontos if p["tipo"] == "Microexpressao"]
        self.assertEqual(len(micro), 1, analisador.atencao.pontos)
        self.assertIn("Medo", micro[0]["descricao"])
        self.assertAlmostEqual(micro[0]["inicio"], 2.0, delta=1 / fps)


if __name__ == "__main__":
    unittest.main()
