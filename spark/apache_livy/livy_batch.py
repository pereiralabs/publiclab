#Lib import
import json, pprint, requests, textwrap, time


def executeBatch(host_url, data, headers):
    #Executes statement
    batches_url = host_url + '/batches'
    r = requests.post(batches_url, data=json.dumps(data), headers=headers)

    #Verifies if statement was executed
    this_statement = host + r.headers['location']
    r = requests.get(this_statement, headers=headers)
    json_data = json.loads(r.text)

    while (json_data['state'] in ['running','waiting']):
        time.sleep(1)
        r = requests.get(this_statement, headers=headers)
        json_data = json.loads(r.text)

    #Returns the statement url
    return this_statement


if __name__ == "__main__":
    #Env definition
    host = 'http://localhost:8998'
    headers = {'Content-Type': 'application/json'}

    #Code creation
    data = {'file': '/media/sf_bitbucket/tmp/SimpleApp.py','className':'com.example.SparkApp'}

    #Batch execution
    print ("Executing batch")
    batchesUrl = executeBatch(host, data, headers)

    #Execution output retrieval    
    r = requests.get(batchesUrl, headers=headers)
    print ("Batch executed. Details:")
    json_data = json.loads(r.text)
    pprint.pprint(json_data['log'])
