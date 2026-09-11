import json

def format_json_output(expression:str, result:float, indent:int=2) -> str:
    if isinstance(result, float) and result.is_integer():
        result_to_dump = int(result)
    else:
        result_to_dump = result
    output = {
        "expression": expression,
        "result": result_to_dump
    }
    return json.dumps(output, indent=indent)