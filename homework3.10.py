# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 09:39:50 2023

@author: ly
"""

class Node:  #定义节点Node类
    def __init__(self,initdata):
        self.data=initdata
        self.next=None
    def getData(self):
        return self.data
    def getNext(self):
        return self.next
    def setData(self,newdata):
        self.data=newdata
    def setNext(self,newnext):
        self.next=newnext
class UnorderedList:  #定义无序列表类，把节点个数作为初始化信息保存在列表头中
    def __init__(self):
        self.head=None
        self.len=0     #初始化表长度
    def add(self,item):   #只有在添加和删除函数中才有列表元素的变化，所以在这两个函数中定义长度的加减即可
        temp=Node(item)
        temp.setNext(self.head)
        self.head=temp
        self.len+=1
    def search(self,item):
        current=self.head
        found=False
        while current!=None and not found:
            if current.getData()==item:
                found=True
            else:
                current=current.getNext()
        return found

    #14.重写remove方法，正确处理待移除结点不在链表中
    def remove_1(self,item):
    #方法一（不推荐，算法时间复杂度高）：先调用search方法遍历链表，查找item
        s=self.search(item)     #调用search方法遍历列表
        if s==False:            #排除元素item不在列表的情况
            print("error:待移除元素不在列表中!")
        else:
            current=self.head
            previous=None
            found=False
            while not found:
                if current.getData()==item:
                    found=True
                else:
                    previous=current
                    current=current.getNext()
            if previous==None:
                self.head=current.getNext()
            else:
                previous.setNext(current.getNext)
            self.len-=1
            
    def remove(self,item):
        #方法二：在循环条件中加入被删除结点不在链表的情况。（推荐）
        current = self.head
        previous = None
        found = False
        while not found and current !=None:
            if current.getData() == item:
                found = True
            else:
                previous = current
                current = current.getNext()   #双指针遍历
        if found:
            if previous == None:
                self.head = current.getNext()     #表头删除
            else:
                previous.setNext(current.getNext())    #中段删除
            self.len -= 1
        else:                               #没找到
            print("error：待移除元素不在列表中！")
            
#18题 实现无需表抽象数据类型剩余方法：append index pop insert
    def append(self,item):   #链表末尾添加元素item
        current = self.head
        temp = Node(item)   #item为数据，实例化为temp结点
        if current==None:    #如果是空链表，添加在头指针后
            self.head=temp
        else:
            while current.getNext() !=None:  #循环找到链表末尾
                current = current.getNext()
            current.setNext(temp)        #链表末尾指针指向temp
        self.len=self.len+1
        return current.getData()

    def index(self,item):    #返回数据为item的结点位置（单链表的第几个结点）
        current = self.head
        found = False
        a=1            #初始化位置变量，结点从1计数
        while current != None and not found:  
            if current.getData() == item:   #找到数据为item的结点，found标记为True，位置为a
                found = True
            else:
                current = current.getNext()  #current指针遍历链表，位置加1
                a=a+1
        if found:
            return a         #返回位置a
        else:
            return "链表不存在item元素"

    def pop(self):    #删除链表末尾结点 
        current = self.head
        prvious = None
        if current == None:
            return "链表空，无法弹出元素!"
        while current.getNext() !=None:  #previous和current两个指针遍历整个链表
            previous = current
            current = current.getNext()
        if previous == None:
            self.head = None
            self.len = self.len-1
            return current.getData()
        else:
            previous.setNext(None)   #设置previous指向None，即删除表最后一个结点
            self.len=self.len-1
            return current.getData() #返回被删除的结点数据项

    def insert(self,position,item):  #在指定位置position插入结点item
        precious = None
        current = self.head
        temp=Node(item)       #插入的结点初始化
        if position > 1:        #position>1，中间插入结点
            for i in range(position): #i的范围没关系，个数对就行
                precious=current
                current=current.getNext()
            precious.setNext(temp)
            temp.setNext(current)
            self.len = self.len + 1
            return current.getData()

        elif position == 1:
            temp.setNext(self.head)
            self.head=temp
            self.len = self.len + 1
            return current.getData()
        else:
            return "位置错误！"


a=UnorderedList()
a.add(1)
a.add(2)
a.add(3)
print(a)
b = a.append(4)
print(b)
print(a.index(2))
print(a.pop())
print(a)
c = a.insert(2,-2)#位置为3时为最后一个元素后添加，current指向为None。
print(a.insert(3,-2))


        
        
        
        
        
        
        
        
        