def REGISTRY="docker.qualityminds.de"
def IMAGE="attention/cist-gcn"
def TAG=BRANCH_NAME.replaceAll('/', '-') + '-latest'

node('docker') {
    stage('checkout') {
        checkout scm
    }

    stage('build') {
        sh "docker build --tag ${REGISTRY}/${IMAGE}:${TAG} ."
    }

    stage('publish') {
        withCredentials([usernamePassword(credentialsId: 'nexus_jenkins_user', passwordVariable: 'pw', usernameVariable: 'user')]) {
            sh "docker login -u ${user} -p ${pw} ${REGISTRY}"
        }
        sh "docker push ${REGISTRY}/${IMAGE}:${TAG}"
    }
}

