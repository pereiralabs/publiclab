# HBASE

Os passos abaixo mostram como instalar o HBASE em um ambiente Debian.

### Instalar Java
``sudo apt install openjdk-8-jre-headless``

### Download HBASE
```
cd opt
mkdir hbase
cd hbase
wget http://ftp.unicamp.br/pub/apache/hbase/stable/hbase-1.4.9-bin.tar.gz
```

### Extract
```tar zxf hbase-1.4.9-bin.tar.gz```

### Set Java Home
```
echo "export JAVA_HOME=/usr" > /etc/profile.d/java.sh
chmod 755 /etc/profile.d/java.sh
source /etc/profile.d/java.sh
```

### Criar diretório de dados
```
mkdir $HOME/cogcladata
mkdir $HOME/cogcladata/hbase
mkdir $HOME/cogcladata/zookeeper
```

### Ajustar conf/hbase-site.xml
```
cd hbase-1.4.9/
vi conf/hbase-site.xml
```

### Colocar dentro da tag CONFIGURATION
```
<configuration>
  <property>
    <name>hbase.rootdir</name>
    <value>file:///root/cogcladata/hbase</value>
  </property>
  <property>
    <name>hbase.zookeeper.property.dataDir</name>
    <value>/root/cogcladata/zookeeper</value>
  </property>
  <property>
    <name>hbase.unsafe.stream.capability.enforce</name>
    <value>false</value>
    <description>
      Controls whether HBase will check for stream capabilities (hflush/hsync).

      Disable this if you intend to run on LocalFileSystem, denoted by a rootdir
      with the 'file://' scheme, but be mindful of the NOTE below.

      WARNING: Setting this to false blinds you to potential data loss and
      inconsistent system state in the event of process and/or node failures. If
      HBase is complaining of an inability to use hsync or hflush it's most
      likely not a false positive.
    </description>
  </property>
</configuration>
```

### Start HBASE
```bin/start-hbase.sh```
