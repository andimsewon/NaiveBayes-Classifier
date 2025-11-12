import csv
import sys
#csv파일 읽기
def readCsv(filePath):
    with open(filePath, 'r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    features = data[0]  # 1st row는 feature로 저장
    rows = data[1:]     # 나머지는 실제 data
    return features, rows

#label frquency count
def calculLabelCount(data, labelCol):
    labelCounts={}
    for row in data:
        label=row[labelCol]
        labelCounts[label] = labelCounts.get(label, 0) + 1
    return labelCounts

#training
def trainNaiveBayes(data, labelCol):
    labelCounts = calculLabelCount(data, labelCol)
    featureCounts = {}

    for row in data:
        label = row[labelCol]

        for col, value in enumerate(row):
            if col == labelCol:
                continue  #label 열 제외
            if col not in featureCounts:
                featureCounts[col] = {}  #해당 col에 대한 딕셔너리 생성
            if value not in featureCounts[col]:
                featureCounts[col][value] = {}  # 해당 val에 대한 딕셔너리 생성
          
            featureCounts[col][value][label] = featureCounts[col][value].get(label, 0) + 1

    #prior 계산
    totalRows = len(data)
    priors = {label: count / totalRows for label, count in labelCounts.items()}
    return priors, featureCounts, labelCounts

#testing
def testNaiveBayes(testRow, priors, featureCounts, labelCounts, totalRows, labelCol):
    results = {}  #label 확률 결과 저장
    for label, priorProb in priors.items():
        prob = priorProb 
        for col, value in enumerate(testRow):
            if col == labelCol:
                continue  #label 열 제외
            if col in featureCounts and value in featureCounts[col]:
                # 해당 특징 값 학습 데이터에 존재하는 경우
                count = featureCounts[col][value].get(label, 0)
                prob *= (count + 1) / (labelCounts[label] + len(featureCounts[col]))  # defending about spartcity: 라플라스 스무딩
            else:
                # 해당 특징 값 학습 데이터에 존재하지 않는 경우
                prob *= 1 / (labelCounts[label] + len(featureCounts[col]))  # defending about spartcity: 라플라스 스무딩
        results[label] = prob

    # MAP rule
    totalProb = sum(results.values())
    ratios = {label: prob / totalProb for label, prob in results.items()}
    ResultLabel = max(ratios, key=ratios.get)
    return ResultLabel, ratios[ResultLabel]

def main():
    #명령어 인자로 구분
    if len(sys.argv)!= 5 or sys.argv[1] != "--train" or sys.argv[3] != "--test":
        print("Commend를 입력하세요: ")
        sys.exit(0)

    trainFile = sys.argv[2]  # train file
    testFile = sys.argv[4]   # test file
    #train
    trainFeatures, trainRows = readCsv(trainFile)
    labelCol = trainFeatures.index("LifestylePrediction") 
    priors, featureCounts, labelCounts = trainNaiveBayes(trainRows, labelCol)
    # test 통해 검증
    testFeatures, testRows = readCsv(testFile)
    for testRow in testRows:
        ResultLabel, ResultRatio = testNaiveBayes(testRow, priors, featureCounts, labelCounts, len(trainRows), labelCol)
        print(f"{ResultLabel} ({round(ResultRatio, 5)})")

#실행
if __name__ == "__main__":
    main()
    print("Succeed!")
