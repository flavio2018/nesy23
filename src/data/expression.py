import numpy as np
import re


class Addition:
    def __init__(self):
        self.operands = 2
        self.used_params = 0
        
    def evaluate(self, values):
        op1, op2 = values
        return op1 + op2
    
    def generate_code(self, codes):
        code_op1, code_op2 = codes
        return "(" + str(code_op1) + "+" + str(code_op2) + ")"


class Subtraction:
    def __init__(self):
        self.operands = 2
        self.used_params = 0
    
    def evaluate(self, values):
        op1, op2 = values
        return op1 - op2

    def generate_code(self, codes):
        code_op1, code_op2 = codes
        return "(" + str(code_op1) + "-" + str(code_op2) + ")"


class Multiplication:
    def __init__(self):
        self.operands = 2
        self.used_params = 0
    
    def evaluate(self, values):
        op1, op2 = values
        return op1 * op2

    def generate_code(self, codes):
        code_op1, code_op2 = codes
        return "(" + str(code_op1) + "*" + str(code_op2) + ")"


class Assignment:
    def __init__(self, used_letters):
        self.operands = 1
        self.used_letters = used_letters
    
    def evaluate(self, values):
        return values[0]
    
    def generate_code(self, codes):
        letter = np.random.randint(len(self.used_letters))
        letter_code = self.used_letters.pop(letter)
        valueless_code = letter_code + '=' + str(codes[0])
        return letter_code, valueless_code


class IfStatement:
    def __init__(self):
        self.operands = 4
        self.geq = True
        self.used_params = 0
        
    def evaluate(self, values):
        res1, res2, op1, op2 = values
        self.geq = True if np.random.rand() > 0.5 else False
        if self.geq:
            if op1 > op2:
                return res1
            else:
                return res2
        else:
            if op1 < op2:
                return res1
            else:
                return res2
        
    def generate_code(self, codes):
        code_res1, code_res2, code_op1, code_op2 = codes
        op = ">" if self.geq else "<"
        return "(" + code_res1 + "if" + code_op1 + op + code_op2 + "else" + code_res2 + ")"


class ForLoop:
    def __init__(self, length):
        self.operands = 1
        self.length = length
        self.used_params = 0
        
    def _set_accumulator(self):
        self.accumulator_code = 'x'
        self.accumulator_value = np.random.randint(10**self.length)
        
    def evaluate(self, values):
        self._set_accumulator()
        self.num_loops = np.random.randint(1, 10)
        accumulator = values[0] 
        for l in range(self.num_loops):
            accumulator += self.accumulator_value
        return accumulator
    
    def generate_code(self, codes):
        return "(x=" + codes[0] + "for[" + str(self.num_loops) + "]" + "x+=" + str(self.accumulator_value) + ")"


def generate_sample(length, nesting, split='train', ops='asmif', steps=False, sample2split=None):
    program_split = ''

    while(program_split != split):
        stack = []
        ops_dict = {
            "a": Addition(),
            "s": Subtraction(),
            "m": Multiplication(),
            "i": IfStatement(),
            "f": ForLoop(length),
        }
        ops_subset = [v for k, v in ops_dict.items() if k in ops]
        program = ''
        intermediate_values = []
        
        for i in range(nesting):
            op = ops_subset[np.random.randint(len(ops_subset))]
            op.used_params = 0
            values = []
            codes = []

            for param in range(op.operands):
                if stack:
                    value, code = stack.pop()
                else:
                    if steps and op.used_params == 0:
                        value = np.random.randint(-10**length+1, 10**length)
                    else:
                        value = np.random.randint(0, 10**length)  # include 1-digit numbers as 2nd operand
                    code = str(value)
                op.used_params += 1        
                values.append(value)
                codes.append(code)
            new_value = op.evaluate(values)
            new_code = op.generate_code(codes)
            stack.append((new_value, new_code))
            intermediate_values.append(new_value)
        final_value, final_code = stack.pop()
        program += final_code

        if sample2split is None:
            program_hash = hash(program)
            if program_hash % 3 == 0:
                program_split = 'train'
            elif program_hash % 3 == 1:
                program_split = 'valid'
            else:
                program_split = 'test'
        else:
            try:
                if ((nesting <= 2) and (length <= 2)):
                    program_split = sample2split[program]
                else:
                    program_split = 'test'
            except KeyError:
                if ((nesting <= 2) and (length <= 2)):
                    program_hash = hash(program)
                    if program_hash % 3 == 0:
                        program_split = 'train'
                    elif program_hash % 3 == 1:
                        program_split = 'valid'
                    else:
                        program_split = 'test'
                else:
                    program_split = 'test'

    solution_steps = get_solution_steps(new_code, intermediate_values)

    if steps:
        return program, str(final_value), solution_steps, [str(v) for v in intermediate_values]
    else:
        return program, str(final_value)


def get_solution_steps(code, values):
    solution_steps = [code]
    for value in values:
        solution_steps.append(re.sub(r'[(][a-z0-9+*\-:=<>\[\] ]+[)]', str(value), solution_steps[-1], count=1))
    return solution_steps