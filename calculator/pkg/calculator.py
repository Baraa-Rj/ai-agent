from collections.abc import Callable


class Calculator:
    def __init__(self) -> None:
        self.operations: dict[str,Callable[[float,float],float]]={
            "+": lambda x,y: x+y,
            "-": lambda x,y: x-y,
            "*": lambda x,y: x*y,
            "/": lambda x,y: x/y if y!=0 else float('inf'),
        }
        self.precedence: dict[str,int] = {
            "+": 1,
            "-": 1,
            "*": 2,
            "/": 2,
        }
        
    def evaluate(self, expression: str)->float|None:
        if not expression or expression.isspace():
            return None
        tokens = expression.strip().split()
        return self._evaluate_infix(tokens)
    
    def _evaluate_infix(self,tokens:list[str])->float:
        values:list[float]=[]
        operators:list[str]=[]
        for token in tokens:
            if token in self.operations:
                while(operators and operators[-1] in self.operations and self.precedence[operators[-1]] >= self.precedence[token]):
                    self._apply_operator(operators, values)
                operators.append(token)
            else:
                try:
                    values.append(float(token))
                except ValueError:
                    raise ValueError(f"Invalid token: {token}")
        while operators:
            self._apply_operator(operators, values)
        if len(values)!=1:
            raise ValueError("Invalid expression")
        return values[0]
    def _apply_operator(self, operators: list[str], values: list[float]) -> None:
        if not operators:
            return
        operator = operators.pop()
        if len(values)<2:
            raise ValueError("Not enough operands")
        b = values.pop()
        a = values.pop()
        values.append(self.operations[operator](a, b))
        

        