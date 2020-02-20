import preprocessing.text.utils as utils


class PreProcessing:
    """
        Essa classe é utilizada na etapa de limpeza de dados, é onde tem todas as funções para limpar dados string
    """

    def __init__(self):
        self.__accents = utils.ACCENTS
        self.__s_accents = utils.S_ACCENTS

    def remove_accent(self, text):
        """
            Essa função troca a vogais que possuem acentuação pela mesma vogal sem o acento.

        :param text: texto que terá os acentos removidos.
        :return: retorna texto sem acentos.
        """

        for i in range(0, len(self.__accents)):
            text = text.replace(self.__accents[i], self.__s_accents[i])

        return text
