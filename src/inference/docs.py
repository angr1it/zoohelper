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
    surgery: yes,
    age": adult,
    rectal_temp: 38.1,
    pulse: 132.0,
    """
)

REQUEST_HELP = inspect.cleandoc(
    """
    Не знаю такой команды. Попробуйте /help, чтобы посмотреть формат ввода.
    """
)

HELLO = inspect.cleandoc(
    """
    Привет. Это бот решения проблемы предсказания здоровья лошади.
    https://www.kaggle.com/competitions/playground-series-s3e22
    Попробуйте /help, чтобы посмотреть возможные команды.
    """
)


def dict_to_doc(d: dict) -> str:
    return json.dumps(d, indent=4)
