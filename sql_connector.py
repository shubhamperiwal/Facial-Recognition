#!/usr/bin/python
import MySQLdb

def selectAll():    
    db = open()
    cur = db.cursor()
    cur.execute("SELECT * FROM user_names")
    results = cur.fetchall()
    cur.close()
    db.close()
    return results

def insert(name):    
    db = open()
    cur = db.cursor()
    num_rows = len(selectAll())

    sql = "INSERT INTO user_names values ({n}, '{nm}')".format(n=num_rows+1, nm=name)
    result = cur.execute(sql)
    cur.close()    
    db.close()
    return num_rows+1

def retrieve(id):    
    db = open()
    cur = db.cursor()

    sql = "select name from user_names where id={i}".format(i=id)
    cur.execute(sql)
    result = cur.fetchone()
    cur.close()    
    db.close()
    return result[0]
    
def open():
    db = MySQLdb.connect(host="localhost",  
                     user="root",         
                     passwd="",  
                     db="facial_recognition")
    return db
        