import sys
import logging
import pymysql
import json

def lambda_handler(event, context):
    #rds settings
    rds_host  = 'host'
    name = 'user'
    password = 'passw'
    db_name = 'db'
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    try:
        conn = pymysql.connect(rds_host, user=name, passwd=password, db=db_name, connect_timeout=5)
    except:
        logger.error("ERROR: Unexpected error: Could not connect to MySql instance.")
        sys.exit()
    
    logger.info("SUCCESS: Connection to RDS mysql instance succeeded")
    
    try:
        str_event = json.dumps(event['Records'][0]['body'])
        str_event = str_event.replace('\\n','')
        str_event = str_event.replace('\\','')
        str_event = str_event.replace(' : ',': ')
    except Exception as e:
        print(e)
        raise e

    with conn.cursor() as cur:
        try:
	    query = "insert into event(insert_ts, event_head, event) values( current_timestamp, '{}', '{}');".format(str_event[:250], str_event) 
            cur.execute(query)
            conn.commit()
        except Exception as e:
            print(e)
            raise e
            
        
