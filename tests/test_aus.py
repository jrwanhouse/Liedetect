"""Revisao da deteccao de cada Action Unit: cada expressao isolada deve gerar so a AU esperada.

Rodar: python -m unittest discover -s tests -v
"""
import unittest

from auxiliares import POSE_FRONTAL, aus_de, girar
from facs import NAO_DETECTAVEIS, NOMES_AU, PONTOS_AU, classificar_emocoes

FORTE = 0.8
CABECA = {51, 52, 53, 54, 55, 56, 57, 58}


def bilateral(base, valor=FORTE):
    return {base + "Left": valor, base + "Right": valor}


class AUsFaciais(unittest.TestCase):
    """Blendshape isolado -> exatamente a AU correspondente."""

    CASOS = {
        1: {"browInnerUp": FORTE},
        2: bilateral("browOuterUp"),
        4: bilateral("browDown"),
        5: bilateral("eyeWide"),
        6: bilateral("cheekSquint"),
        7: bilateral("eyeSquint"),
        9: bilateral("noseSneer"),
        10: bilateral("mouthUpperUp"),
        12: bilateral("mouthSmile"),
        14: bilateral("mouthDimple"),
        15: bilateral("mouthFrown"),
        17: {"mouthShrugLower": FORTE},
        18: {"mouthPucker": FORTE},
        20: bilateral("mouthStretch"),
        22: {"mouthFunnel": FORTE},
        23: bilateral("mouthPress"),
        28: {"mouthRollUpper": FORTE},
        29: {"jawForward": FORTE},
        34: {"cheekPuff": FORTE},
    }

    def test_cada_au_isolada(self):
        for au, blendshapes in self.CASOS.items():
            with self.subTest(au=au):
                self.assertEqual(set(aus_de(blendshapes)), {au})

    def test_au16_abaixa_o_labio_e_separa_os_labios(self):
        self.assertEqual(set(aus_de(bilateral("mouthLowerDown"))), {16, 25})

    def test_abertura_da_boca_25_26_27(self):
        self.assertEqual(set(aus_de({"jawOpen": 0.22})), {25})       # so labios separados
        self.assertEqual(set(aus_de({"jawOpen": 0.45})), {25, 26})   # queda da mandibula
        self.assertEqual(set(aus_de({"jawOpen": 0.90})), {25, 27})   # estiramento: 27 no lugar de 26

    def test_au30_mandibula_lateral_tem_lado(self):
        aus = aus_de({"jawLeft": FORTE})
        self.assertEqual(set(aus), {30})
        self.assertEqual(aus[30].lado, "L")

    def test_neutro_apos_calibracao_nao_tem_aus(self):
        self.assertEqual(aus_de({}), {})


class AU6eAU7(unittest.TestCase):
    """O modelo quase nao ativa cheekSquint: o olho apertado e AU6 no sorriso e AU7 fora dele."""

    def test_sorriso_com_olhos_apertados_e_duchenne(self):
        aus = aus_de(bilateral("mouthSmile") | bilateral("eyeSquint"))
        self.assertEqual(set(aus), {6, 12})
        self.assertEqual(set(classificar_emocoes(aus)), {"Felicidade"})

    def test_olhos_apertados_sem_sorriso_e_au7(self):
        self.assertEqual(set(aus_de(bilateral("eyeSquint"))), {7})


class Palpebras(unittest.TestCase):
    """AU43/45/46 e a supressao de AU7 e do olhar enquanto as palpebras fecham."""

    # ao fechar os olhos o modelo tambem sobe eyeSquint e eyeLookDown (verificado em foto real)
    FECHANDO = bilateral("eyeBlink", 0.9) | bilateral("eyeSquint", 0.9) | bilateral("eyeLookDown", 0.6)

    def test_piscada_e_so_au45(self):
        self.assertEqual(set(aus_de(self.FECHANDO, frames=3)), {45})

    def test_olhos_fechados_por_mais_de_meio_segundo_sao_au43(self):
        self.assertEqual(set(aus_de(self.FECHANDO, frames=20)), {43})

    def test_piscadela_tem_lado_e_nao_gera_au7(self):
        aus = aus_de({"eyeBlinkLeft": 0.9, "eyeSquintLeft": 0.9})
        self.assertEqual(set(aus), {46})
        self.assertEqual(aus[46].lado, "L")

    def test_palpebra_descendo_nao_vira_olhar_para_baixo(self):
        meio_fechado = bilateral("eyeBlink", 0.5) | bilateral("eyeLookDown", 0.6)
        self.assertNotIn(64, aus_de(meio_fechado))


class Olhar(unittest.TestCase):
    """AU61-64; no padrao do MediaPipe 'Left' e o olho esquerdo do sujeito (verificado em foto real)."""

    def test_direcoes(self):
        casos = {
            61: {"eyeLookOutLeft": FORTE, "eyeLookInRight": FORTE},  # olhar para a esquerda do sujeito
            62: {"eyeLookInLeft": FORTE, "eyeLookOutRight": FORTE},
            63: bilateral("eyeLookUp"),
            64: bilateral("eyeLookDown"),
        }
        for au, blendshapes in casos.items():
            with self.subTest(au=au):
                self.assertEqual(set(aus_de(blendshapes)), {au})


class Lateralidade(unittest.TestCase):
    def test_au_unilateral_recebe_lado(self):
        aus = aus_de({"browOuterUpLeft": FORTE})
        self.assertEqual((set(aus), aus[2].lado), ({2}, "L"))
        aus = aus_de({"mouthSmileRight": FORTE})
        self.assertEqual((set(aus), aus[12].lado), ({12}, "R"))
        self.assertEqual(set(classificar_emocoes(aus)), {"Desprezo"})

    def test_bilateral_nao_tem_lado(self):
        self.assertEqual(aus_de(bilateral("mouthSmile"))[12].lado, "")


class Cabeca(unittest.TestCase):
    """AU51-58 pela geometria dos landmarks, relativa a pose da calibracao."""

    def cabeca(self, pontos, blendshapes=None):
        return set(aus_de(blendshapes, pontos=pontos)) & CABECA

    def test_giro(self):
        x, y = POSE_FRONTAL[1]
        # nariz para a direita da imagem = cabeca virada para a esquerda do sujeito
        self.assertEqual(self.cabeca({1: (x + 0.05, y)}), {51})
        self.assertEqual(self.cabeca({1: (x - 0.05, y)}), {52})

    def test_para_cima_e_para_baixo(self):
        x, y = POSE_FRONTAL[1]
        self.assertEqual(self.cabeca({1: (x, y - 0.03)}), {53})
        self.assertEqual(self.cabeca({1: (x, y + 0.03)}), {54})

    def test_boca_aberta_nao_vira_cabeca_para_cima(self):
        # o queixo desce com a mandibula; antes isso gerava AU53 falsa
        aus = aus_de({"jawOpen": 0.9}, pontos={152: (0.5, 0.86)})
        self.assertIn(27, aus)
        self.assertEqual(set(aus) & CABECA, set())

    def test_inclinacao_lateral(self):
        # +12 graus na tela (anti-horario) levanta o olho esquerdo do sujeito: inclina para a direita
        self.assertEqual(self.cabeca(girar(POSE_FRONTAL, 12)), {56})
        self.assertEqual(self.cabeca(girar(POSE_FRONTAL, -12)), {55})

    def test_frente_e_tras(self):
        escala = lambda k: {i: (0.5 + (x - 0.5) * k, 0.5 + (y - 0.5) * k) for i, (x, y) in POSE_FRONTAL.items()}
        self.assertEqual(self.cabeca(escala(1.25)), {57})
        self.assertEqual(self.cabeca(escala(0.80)), {58})


class Intensidade(unittest.TestCase):
    def test_intensidade_cresce_com_o_sinal(self):
        letras = [aus_de({"browInnerUp": v})[1].intensidade for v in (0.4, 0.6, 0.8, 1.0)]
        self.assertEqual(letras, sorted(letras))
        self.assertEqual((letras[0], letras[-1]), ("A", "E"))

    def test_au26_alcanca_intensidade_maxima_antes_da_au27(self):
        self.assertIn(aus_de({"jawOpen": 0.62})[26].intensidade, "DE")


class Cadastro(unittest.TestCase):
    """Coerencia das tabelas de AUs."""

    FACS = set(range(1, 47)) - {3, 40}  # AUs faciais do FACS (3 e 40 nao existem)

    def test_todas_as_aus_do_facs_estao_classificadas(self):
        self.assertEqual(self.FACS - set(NOMES_AU) - set(NAO_DETECTAVEIS), set())
        self.assertEqual(set(NOMES_AU) & set(NAO_DETECTAVEIS), set())

    def test_toda_au_detectavel_tem_ponto_no_rosto(self):
        self.assertEqual(set(NOMES_AU), set(PONTOS_AU))


if __name__ == "__main__":
    unittest.main()
