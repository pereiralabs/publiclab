import sys
import logging
import pymysql

def lambda_handler(event, context):
    #rds settings
    rds_host  = 'plabs-reb-rds-mysql.crj33fqtdwng.us-east-1.rds.amazonaws.com'
    name = 'plabs'
    password = 'plabs2019'
    db_name = 'dw'
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    try:
        conn = pymysql.connect(rds_host, user=name, passwd=password, db=db_name, connect_timeout=5)
    except:
        logger.error("ERROR: Unexpected error: Could not connect to MySql instance.")
        sys.exit()
    
    logger.info("SUCCESS: Connection to RDS mysql instance succeeded")
    def handler(event, context):
        """
        This function fetches content from mysql RDS instance
        """
    
        item_count = 0
    
        with conn.cursor() as cur:
            cur.execute("create table log ( pk_id  int NOT NULL, log_date date NOT NULL, log_text varchar(256) null);")  
            cur.execute('insert into log (pk_id, log_date, log_text) values (1, current_timestamp, \'teste\');')
            conn.commit()
            #cur.execute("select * from Employee3")
            #for row in cur:
            #    item_count += 1
            #    logger.info(row)
            #    #print(row)
         #conn.commit()
    
        #return "Added %d items from RDS MySQL table" %(item_count)
