import pulp
from pulp import LpProblem, LpVariable, LpMaximize

class schedule :

    # 발전소 갯수, 건물 개수
    PV_count=1
    L_count=1
    hours=24

    # 계산 요금
    low_price = 70.5
    middle_price = 114.0
    high_price = 176.9

    electricity_prices = [middle_price] * 24  # 기본 비용으로 배열 초기화
    # BESS 관련 변수
    BESS_max = 750 #BESS 용량
    SOC_min = 0.2 * BESS_max  # 최소 SOC
    SOC_max = 0.8 * BESS_max  # 최대 SOC
    SOC_initial = 0.5 * BESS_max  # 초기 SOC
    efficiency = 0.9  # BESS 충/방전 효율

    #가격 계산 
    def __init__(self):
        # 22:00~08:00 시간대에 70.5 적용
        for hour in range(22, 24):
            self.electricity_prices[hour] = self.low_price
        for hour in range(0, 8):
            self.electricity_prices[hour] = self.low_price

        # 11:00~12:00 시간대에 176.9 적용
        for hour in range(11, 13):
            self.electricity_prices[hour] = self.high_price

        # 12:00~18:00 시간대에 176.9 적용
        for hour in range(12, 19):
            self.electricity_prices[hour] = self.high_price

    def price_sum(self,P_grid_predict) :
        # 총 사용 전기량
        low_price_wart = 0
        middle_price_wart = 0
        high_price_wart = 0

        for t in range(self.hours):
            if t in range(22, 24) or t in range(0, 8):
                low_price_wart += pulp.value(P_grid_predict[t])
            elif t in range(11, 13) or t in range(12, 19):
                middle_price_wart += pulp.value(P_grid_predict[t])
            else :
                high_price_wart +=pulp.value(P_grid_predict[t])
            
        # 교육용(을) 계약전력 10000Kw
        base_price_per_day = 10000 * 6980 / 30

        total_price_wart = low_price_wart * 24 + middle_price_wart * 24 + high_price_wart * 24
        total_price = (low_price_wart * self.low_price + middle_price_wart * self.middle_price + high_price_wart * self.high_price) * 24

        price_prev_tax = base_price_per_day +total_price/30 + (total_price_wart*14)/30  #연료비조정액, 기후환경요금
        price = price_prev_tax*1.137
        return price
    
    def schedule_optimizer(self,PV_sum_list, L_sum_list) :
        prob = pulp.LpProblem("Optimal_Energy_Management", pulp.LpMinimize)

        P_charge = pulp.LpVariable.dicts("P_charge", [(t, i) for t in range(self.hours) for i in range(self.PV_count)], lowBound=0)
        P_discharge = pulp.LpVariable.dicts("P_discharge", [(t, i) for t in range(self.hours) for i in range(self.PV_count)], lowBound=0)
        
        # 구매해야되는 전기량
        P_grid = pulp.LpVariable.dicts("P_grid", [t for t in range(self.hours)], lowBound=0)

        SOC = pulp.LpVariable.dicts("SOC", [t for t in range(self.hours)], lowBound=0, upBound=750)

        #Object function
        prob += pulp.lpSum(P_grid[t] * self.electricity_prices[t] for t in range(self.hours))

        # 제약 조건
        for t in range(self.hours):
            for i in range(self.PV_count):
                prob += L_sum_list[i][t] - P_discharge[(t, i)] == P_grid[t]
                prob += P_charge[(t, i)] <= PV_sum_list[i][t]
                if t>=1 :
                    prob += SOC[t] == SOC[t - 1] + (P_charge[(t, i)] - P_discharge[(t, i)]) * self.efficiency
                else :
                    prob += SOC[t] == self.SOC_initial + (P_charge[(t, i)] - P_discharge[(t, i)]) * self.efficiency
                prob += SOC[t] >= self.SOC_min
                prob += SOC[t] <= self.SOC_max
        
        prob += SOC[0] == self.SOC_initial
        prob += SOC[23] == self.SOC_initial
        
        prob.solve(pulp.PULP_CBC_CMD(msg=False))  # 로그 메시지 출력 안함

        for t in range(self.hours):
            print(f"Hour {t}:")
            for i in range(self.PV_count):
                print(f"P_charge[{i}] =", pulp.value(P_charge[(t, i)]))
                print(f"P_discharge[{i}] =", pulp.value(P_discharge[(t, i)]))
            print("SOC =", pulp.value(SOC[t]))
            print("------")
        print("확인")

        data = [pulp.value(SOC[k]) for k in range(self.hours)]


        return self.price_sum(P_grid) ,data