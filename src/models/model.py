from inference.errors import WrongParamsError


def check_params(data: dict) -> bool:
    if "a" not in data:
        raise WrongParamsError
    if "b" not in data:
        raise WrongParamsError
    return True


def start(data: dict) -> dict:
    check_params(data)
    try:
        return {"result": int(data["a"]) + int(data["b"])}
    except Exception:
        raise WrongParamsError
