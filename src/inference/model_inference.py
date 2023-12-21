import re

from models.model import Model
from inference.errors import ParseError, InternalError, WrongParamsError


def parse_input(features: dict, message: str) -> dict:
    try:
        message = re.sub("^\s+|\n|\r|\s+$", "", message)
        message = message.replace(" ", "")

        params = {}
        for pair in message.split(","):
            if not pair or pair == '':
                continue

            key, value = pair.split(":", 1)
            if key in params:
                raise Exception
            if key not in features:
                continue
            if features[key] == "numeric":
                params[key] = float(value)
            else:
                params[key] = str(value).strip()

        return params
    except Exception:
        raise ParseError


def prepare_output(result) -> str:
    return f'Результат: {str(result)}'


def process(model: Model, message: str) -> str:
    try:
        data = parse_input(model.get_features_dict(), message)
        result = model.predict(data)
        return prepare_output(result)
    except (ParseError, WrongParamsError):
        return "Возникла ошибка парсинга, либо введены неверные параметры."
    except InternalError:
        return "Внутренняя ошибка. Обратитесь к администраторам."
