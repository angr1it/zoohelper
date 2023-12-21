import inspect
import json

HELP_DOC = inspect.cleandoc(
    """
    /process -- ввод данных для расчета;
    /features -- вывод словаря с описанием возможных значений;
    """
)

PROCESS_DOC = inspect.cleandoc(
    """
    Введите переменные в формате key: value через запятую.
    Например:
    "surgery": "yes",
    "age": "adult",
    "rectal_temp": 38.1,
    "pulse": 132.0,
    """
)


def dict_to_doc(d: dict) -> str:
    return json.dumps(d, indent=4)
