from sklearn.metrics import accuracy_score, precision_score, classification_report
from classifier import output
import numpy as np

output_true= np.array([[0, 1, 1, 0, 0, 1, 1, 0, 1, 1], 
                      [1, 1, 0, 1, 1, 0, 0, 1, 1, 0]])

# output은 손실 계산하고 학습하기 위해 계산 그래프에 연결되어 있음. sklearn에 가중치 업데이트 내용을 모름!
# 따라서 해당 연결고리를 끊어줘야 함
output= output.detach().cpu().numpy()

output= (output>= 0.5).astype(int)
accuracy= accuracy_score(output_true, output)

# 정밀도 precision_macro 계산
precision_macro= precision_score(output_true, output, average= 'macro')

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision (Macro Avg): {precision_macro:.3f}")

# 상세 보고서 출력
report= classification_report(output_true, output)
print("Classification Report\n", report)