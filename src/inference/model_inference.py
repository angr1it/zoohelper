import re

from models import model
from inference.errors import ParseError, InternalError, WrongParamsError


def parse_input(input: str) -> dict:
    try:
        input = re.sub("^\s+|\n|\r|\s+$", "", input)
        input = input.replace(" ", "")

        params = {}
        for pair in input.split(","):
            key, value = pair.split(":", 1)
            if key in params:
                raise Exception
            params[key] = value

        return params
    except Exception:
        raise ParseError


def prepare_output(result: dict) -> str:
    return f'Сумма: {result["result"]}'


def process(input: str) -> str:
    try:
        data = parse_input(input)
        result = model.start(data)
        return prepare_output(result)
    except (ParseError, WrongParamsError):
        return "Возникла ошибка парсинга, либо введены неверные параметры."
    except InternalError:
        return "Внутренняя ошибка. Обратитесь к администраторам."
