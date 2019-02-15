import json, pprint, requests, textwrap,time


def createSparkSession(host, headers, sessionType):
    #Starts session
    r = requests.post(host + '/sessions', data=json.dumps(sessionType), headers=headers)

    #Verifies if session is ready
    session_url = host + r.headers['location']
    r = requests.get(session_url, headers=headers)
    json_data = json.loads(r.text)

    while (json_data['state'] == 'starting'):
        time.sleep(1)
        r = requests.get(session_url, headers=headers)
        json_data = json.loads(r.text)

    #Returns the session url
    return session_url


def executeStatement(session_url, data, headers):
    #Executes statement
    statements_url = session_url + '/statements'
    r = requests.post(statements_url, data=json.dumps(data), headers=headers)

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
    #Connect to Spark
    print ("Starting session")
    host = 'http://localhost:8998'
    headers = {'Content-Type': 'application/json'}
    sessionType = {'kind':'pyspark'}
    sessionUrl = createSparkSession(host, headers, sessionType)

    #Code creation
    data = {
      'code': textwrap.dedent("""
        import random
        NUM_SAMPLES = 100000
        def sample(p):
          x, y = random.random(), random.random()
          return 1 if x*x + y*y < 1 else 0

        count = sc.parallelize(range(0, NUM_SAMPLES)).map(sample).reduce(lambda a, b: a + b)
        print (4.0 * count / NUM_SAMPLES)
        """)
    }

    #Code submition
    print ("Submitting statement")
    statementsUrl = executeStatement(sessionUrl, data, headers)

    #Output status
    r = requests.get(statementsUrl, headers=headers)
    print ("Statement executed. Details:")
    json_data = json.loads(r.text)
    pprint.pprint(json_data['output'])

    #Session deletion
    r = requests.delete(sessionUrl, headers=headers)
