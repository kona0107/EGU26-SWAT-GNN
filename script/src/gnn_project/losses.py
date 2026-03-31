import torch
import torch.nn as nn

class MaskedMSELoss(nn.Module):
    """
    위성 데이터의 결측(구름)이나 현장 데이터의 듬성한 샘플링 한계를 극복하기 위한
    Mask 기반 오차 전파(Supervision) 구조 적용.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        # reduction='none'으로 설정해 요소 개별 단위의 MSE를 얻음
        self.mse = nn.MSELoss(reduction='none')
        self.eps = eps

    def forward(self, preds, targets, masks):
        """
        :param preds: [B, N, 1] - 신경망 예측값
        :param targets: [B, N, 1] - 위성 관측 기반 정답(Chl-a 등)
        :param masks: [B, N, 1] - 마스크 (1=신뢰가능한 위성관측/측정소 맵핑 부분, 0=사용불가)
        :return: 마스킹을 통과한 유효 샘플들의 평균 Loss 값
        """
        # 1. 일단 마스킹 무관하게 전체 [B, N, 1] 차원에서의 오차 연산 
        unmasked_loss = self.mse(preds, targets)  
        
        # 2. 값 0(결측/낮은 신뢰도) 인 부분을 곱소거 처리하여 모델 가중치 학습에 방해되지 않도록 차단
        masked_loss = unmasked_loss * masks
        
        # 3. 유효성분 (값이 살아남은 텐서 요소) 합산 추출
        sum_loss = torch.sum(masked_loss)
        
        # 4. 실질적인 관측 데이터 수
        num_valid_samples = torch.sum(masks)
        
        # 5. 유효샘플 만을 모수로 잡아 평균오차 산출 / 분모가 0이 되는 ZeroDivision 차단(eps)
        mean_loss = sum_loss / (num_valid_samples + self.eps)
        
        return mean_loss
