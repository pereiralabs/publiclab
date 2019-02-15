import json
import psycopg2

def lambda_handler(event, context):

    db_host='plabs-reb-rds-postgres.crj33fqtdwng.us-east-1.rds.amazonaws.com'
    db_port = 5432
    db_name = "dw"
    db_user = "plabsreb"
    db_pass = "plabsreb2019"
    db_table = "test"
    
    
    def make_conn():
        conn = None
        try:
            conn = psycopg2.connect("dbname='%s' user='%s' host='%s' password='%s'" % (db_name, db_user, db_host, db_pass))
        except:
            print("I am unable to connect to the database")
        return conn
    
    
    def fetch_data(conn, query):
        result = []
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        cursor.close()
    
        #raw = cursor.fetchall()
        #for line in raw:
        #    result.append(line)
    
        #return result
        
    myLog = 'test log'
    query_cmd = "insert into test(insert_date,log) values(current_timestamp,'%s');" % (myLog)

    # get a connection, if a connect cannot be made an exception will be raised here
    conn = make_conn()

    result = fetch_data(conn, query_cmd)
    conn.close()

    return result
