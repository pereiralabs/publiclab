# HBASE

Os passos abaixo devem ser executados na mesma máquina onde o HBASE foi instalado no procedimento anterior ```hbase_install.md```

### Acesse o HBASE
```./bin/hbase shell```

### Crie uma tabela
```create 'test', 'cd'```

### Verifique a tabela
```describe 'test'```

### Volte para o shell do Linux para importar os dados
```
./bin/hbase org.apache.hadoop.hbase.mapreduce.ImportTsv
  -Dimporttsv.separator='|' \ -Dimporttsv.columns=HBASE_ROW_KEY,cd:CPF,cd:CEP,cd:NUM,cd:FG_END_NET,\
  cd:FG_END_CLAROMOVEL,cd:FG_END_CLAROFIXO,cd:FG_END_CLAROTV,cd:FG_END_EBT,\
  cd:FG_END_MERCADO,cd:FG_END_BVS,cd:FG_CEP_ERB_RESIDENCIA,cd:FG_CEP_ERB_COMERCIAL,\
  cd:FG_CEP_ERB_TRANSITO,cd:FG_CEP_ERB_FIM_SEMANA \
  test \
  /tmp/data/
```

### Verifique a inserção
```
./bin/hbase shell
scan 'test'
```
