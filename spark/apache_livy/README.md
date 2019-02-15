# Apache Livy

Apache Livy is an open-source project that enables RESTful communication between the Spark Cluster and Spark Clients.

Clients can submit batches and statements through a REST API and Spark will receive, process and return the result through that same API.

The process installation and the official documentation can be found at:
`` https://livy.incubator.apache.org/ ``

### Configuration

I've kept most of the default configuration for the development environment. The only changes were:

  ```
  #$LIVY_HOME/conf/livy.conf

  # How long a finished session state should be kept in LivyServer for query.
  livy.server.session.state-retain.sec = 1800s

  # List of local directories from where files are allowed to be added to user sessions. By
  # default it's empty, meaning users can only reference remote URIs when starting their
  # sessions.
  livy.file.local-dir-whitelist = /tmp/  

  ```
