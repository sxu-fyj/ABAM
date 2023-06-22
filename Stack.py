# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 17:44:29 2023

@author: ly
"""

class Stack:
    ############################
    #默认的初始化栈的函数
    ############################
    def __init__(self):
        self.items = []
    #############################
    #判断栈是否为空的函数
    ############################
    def isEmpty(self):
        return self.items == []
    ############################
    #元素进栈的函数
    ############################
    def push(self,item):
        self.items.append(item)
    ############################
    #元素出栈的函数
    ############################
    def pop(self):
        return self.items.pop()
    #############################
    #查看栈顶元素
    #############################
    def peek(self):
        return self.items[len(self.items)-1]
    #############################
    #依次访问栈中元素的函数
    #############################
    def stackTraverse(self):
        if self.isEmpty():
            print("栈为空")
            return
        else:
            for i in range(len(self.items)-1,-1,-1):#从变量i=0开始到i=top为止
                print(self.items[i],end='  ')
    ############################
    #输出当前栈长度的函数
    ############################
    def size(self):
        if self.isEmpty():
            print("栈为空")
            return
        else:
            return len(self.items)
    ###############################
    #由用户输入元素将其进栈的函数
    ###############################
    def create_stack_by_input(self):
        data=input("请输入元素(继续输入请按回车，结束输入请按“#”：")
        while data!='#':
            self.push(data)
            data=input("请输入元素：")
    ###############################
    #利用栈实现后序表达式的计算
    ###############################            
def postfixEval(postfixExpr):
    operandStack = Stack()
    tokenList = postfixExpr.split()
    
    for token in tokenList:
        if token in "0123456789":
            operandStack.push(int(token))
        else:
            operand2 = operandStack.pop()
            operand1 = operandStack.pop()
            result = doMath(token, operand1, operand2)
            operandStack.push(result)
    return operandStack.pop()
    
def doMath(op,op1,op2):
    if op == "*":
        return op1*op2
    elif op == "/":
        return op1/op2
    elif op == "+":
        return op1+op2
    else:
        return op1-op2
    
# =============================================================================
# s = Stack()
# s.create_stack_by_input()
# print("栈内的元素为：",end='')
# s.stackTraverse()
# =============================================================================
result = postfixEval("2 3 * 4 + ")
print(result)