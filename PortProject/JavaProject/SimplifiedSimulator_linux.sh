#!/bin/sh
# 
# Run AnyLogic Experiment
# 
DIR_BACKUP_XJAL=$(pwd)
SCRIPT_DIR_XJAL=$(dirname "$0")
cd "$SCRIPT_DIR_XJAL"
chmod +x chromium/chromium-linux64/chrome
java -cp model.jar:lib/MaterialHandlingLibrary.jar:lib/ProcessModelingLibrary.jar:lib/RoadTrafficLibrary.jar:lib/mysql-connector-java-5.1.39-bin.jar:lib/java-json.jar:lib/bin:lib/jts-core-1.16.0.jar:lib/com.anylogic.engine.jar:lib/com.anylogic.engine.nl.jar:lib/com.anylogic.engine.sa.jar:lib/sa/com.anylogic.engine.sa.web.jar:lib/sa/executor-basic-8.3.jar:lib/sa/ioutil-8.3.jar:lib/sa/spark/commons-codec-1.10.jar:lib/sa/spark/jackson-annotations-2.8.5.jar:lib/sa/spark/jackson-core-2.8.5.jar:lib/sa/spark/jackson-databind-2.8.5.jar:lib/sa/spark/jackson-datatype-jsr310-2.8.5.jar:lib/sa/spark/javax.servlet-api-3.1.0.jar:lib/sa/spark/jetty-client-9.4.8.v20171121.jar:lib/sa/spark/jetty-http-9.4.8.v20171121.jar:lib/sa/spark/jetty-io-9.4.8.v20171121.jar:lib/sa/spark/jetty-security-9.4.8.v20171121.jar:lib/sa/spark/jetty-server-9.4.8.v20171121.jar:lib/sa/spark/jetty-servlet-9.4.8.v20171121.jar:lib/sa/spark/jetty-servlets-9.4.8.v20171121.jar:lib/sa/spark/jetty-util-9.4.8.v20171121.jar:lib/sa/spark/jetty-webapp-9.4.8.v20171121.jar:lib/sa/spark/jetty-xml-9.4.8.v20171121.jar:lib/sa/spark/jsch-0.1.54.jar:lib/sa/spark/slf4j-api-1.7.21.jar:lib/sa/spark/spark-core-2.7.2.jar:lib/sa/spark/websocket-api-9.4.8.v20171121.jar:lib/sa/spark/websocket-client-9.4.8.v20171121.jar:lib/sa/spark/websocket-common-9.4.8.v20171121.jar:lib/sa/spark/websocket-server-9.4.8.v20171121.jar:lib/sa/spark/websocket-servlet-9.4.8.v20171121.jar:lib/sa/util-8.3.jar:lib/database/querydsl/querydsl-core-4.2.1.jar:lib/database/querydsl/querydsl-sql-4.2.1.jar:lib/database/querydsl/querydsl-sql-codegen-4.2.1.jar:lib/database/querydsl/guava-18.0.jar -Xmx8192m simplifiedSimulator.CustomExperiment $*
cd "$DIR_BACKUP_XJAL"
