import numpy as np
import pandas as pd
from sklearn import metrics

#============================================================================================================

class World():
    """Class World - Макромир, который задает начало отсчета времени,
        законы макроэкономики, ограничения регуляторов/ЦБ и остальное окружение.
        Так же здесь описывается поведение клиента, которое может зависеть от внешних факторов.

        dod_migration - матрица вероятностей переходов по просрочкам
    """

    def __init__(self, seed = 0):
        self.seed = seed
        self.Time = 0
        self.Population_Mu = 0
        self.Population_Sigma = 1
        self.Fate_cutoff = 0.3
        #                             0    1+   31+   61+   91+   WOF
        self.dod_migration = np.array([[0.95, 0.05, 0.00, 0.00, 0.00, 0.00], #  0 
                                       [0.90, 0.05, 0.05, 0.00, 0.00, 0.00], #  1+
                                       [0.10, 0.05, 0.05, 0.80, 0.00, 0.00], # 31+
                                       [0.05, 0.05, 0.05, 0.05, 0.80, 0.00], # 61+
                                       [0.01, 0.01, 0.02, 0.02, 0.04, 0.90], # 91+
                                       [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]  # WOF
            #                          [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]  # TODO - добавить досрочное и частичнодочсрочное погашение кредита 
                         ])
        assert self.dod_migration.shape[0] == self.dod_migration.shape[1] # проверка на квадратность
        assert [i.sum() for i in self.dod_migration] == [1 for i in range(len(self.dod_migration))] # 1 in sum of row
        print('test - ', [i.sum() for i in self.dod_migration])

    def print_current_reality(self):
        print('Welcome to the real world')

    # возвращаем предопределенный скор, который неизменен    
    def get_god_score(self, N=1):
        np.random.seed(self.seed)
        
        # предопределенный скор, который пытаемся определить моделью
        score = np.random.normal(self.Population_Mu,
                                 self.Population_Sigma,
                                 N
                                )
        # предопределенная дефолтность - определяет исход контракта
        # 0 - контракт будет закрыт погашением
        # 1 - контракт будет закрыт списанием
        fate  = [1 if scr > self.Fate_cutoff else 0 for scr in score]
        self.seed += 1
        return [score, fate]

#============================================================================================================

class Bank_DS():
    """Class DS - Здесь будут жить фишки отдела DS
    """

    def __init__(self):
        self.N_grade = 16 # Кол-во рейтингов по умолчанию.
        print('Hello DS!')

    # Функция сигмоиды
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    # Расчет Gini 
    def gini(self, df, target='BAD_FLAG', score='SCORE', plot=False):
        fpr, tpr, _ = metrics.roc_curve(df[target], df[score])
        gini = 2 * metrics.auc(fpr, tpr) - 1
        if plot:
            plt.figure(figsize = (5, 5))
            plt.title('ROC')
            plt.plot(fpr, tpr, label = 'GINI = %0.4f'%gini, color = 'green')
            plt.legend(loc = 'lower right') #, fontsize=10)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('Ture Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()
        return gini

    # Применение калибровки
    def calibration(self, Score, B0, B1):
        return 1/(1+np.exp(-B0-B1*Score))

    # Функция границы мастер шкалы
    def func_of_threshold(self, i):
        return 2**((-self.N_grade+i)/2)
   
    # Расчет рейтинга. Можно было бы вывести обратную функцию от границы, но, возможно, это не лучшее решение.
    def grade(self, pd, ):
        for i in range(1,17):
            R_pd = self.func_of_threshold(i)
            if pd < R_pd:
                return i

    # Распечатать границы мастер шкалы
    def print_master_scale(self):
        print('Grage Right_value')
        for g in range(1, self.N_grade+1):
            print('%5s %0.3g'%(g, self.func_of_threshold(g)))

    # Создание модели.
    # Модель создается на выданных кредитах подбором значений зашумления предопределенного скора.
    # Здесь подбираются шум таким образом, чтобы качество (gini) модели соответствовало заданому значению.
    def create_model(self, name, gini, portfolio, tto_period):
        return None

#============================================================================================================

class DWH_DB():
    """Class DWH - база данных
    """
    def __init__(self):
        self.LI = pd.DataFrame(columns = ['CNTR_ID',
                                          'SD',
                                          'DOD_ID',
                                          'MOB',
                                          'WRTOFF_ID',
                                          'CLOSED_ID'
                                          ])
        self.DMContract = pd.DataFrame(columns = ['CNTR_ID',
                                                  'ISSUE_DT',
                                                  'WRTOFF_DT',
                                                  'CLOSED_DT',
                                                  'AMOUNT',
                                                  'DURATION',
                                                  'IR',
                                                  'TARIFF'
                                                 ])
        
#============================================================================================================

class Tariff():
    """Class Tariff - здесь опишем тарифы.
    """

    def __init__(self, name, IR = 0.12, DUR = 24, TypePlan = 'Annuity'):
        self.name = name          # Название тарифного плана
        self.IR = IR              # Годовая процентная ставка
        self.DUR = DUR            # Срок
        self.TypePlan = TypePlan  # Тип кредита. Для использования требутся описание плана : Annuity - аннуитетные платежи. 
        self.PlanParam = {}       # Справочник, для описания различных типов/подходов кредита  
        self.plan()               # Заполним график платежей

    # создание графика платежей
    def plan(self):
        PP = self.PlanParam
        if self.TypePlan == 'Annuity':
            i = self.IR/12 # Ставка в месяц
            n = self.DUR   # Срок в месяцах
            d = (1+i)**n   # вспомогательная переменная
            K = i*d/(d-1)  # Коэффициент аннуитета
            PP['K'] = K    # Коэффициент аннуитета
            PP['MD'] =  1/i- n/(d-1)/(1+i) # Модифицированная дюрация
            # опишем график платежей: 
            PP[1] = np.array([1+i, # Долг на момент оплаты
                              K,   # платеж
                              K-i, # долг (часть платежа)
                              i,   # проценты (часть платежа)
                              K-i, # кумулятив по оплаченному долгу
                              i,   # кумулятив по оплаченным процентам
                              K,   # кумулятив по оплатам
                             ])
            for j in range(2, n+1):
                last_EAD = (PP[j-1][0]-K) 
                PP[j] = np.array([last_EAD*(1+i), K , K-last_EAD*i, last_EAD*i, K-last_EAD*i + PP[j-1][4], last_EAD*i + PP[j-1][5], K*j])

    # Краткая инфа о тарифном плане
    def info(self):
        if self.TypePlan == 'Annuity':
            print('Name: %s TypePlan = %s IR = %s Duration = %i MD = %g'%(self.name,
                                                                                  self.TypePlan,
                                                                                  self.IR,
                                                                                  self.DUR,
                                                                                  self.PlanParam['MD']))

    # Нарисовать график для заданной суммы
    def print_plan(self, amount = 1):
        if self.TypePlan == 'Annuity':
            print(' N', ['%10s'%c for c in ['Долг+проц', 'Платеж', 'Плат(долг)', 'Плат(проц)', 'Кум долг', 'Кум проц', 'Кум оплаты']])
            for i in range(1, self.DUR+1):
                print('%02i'%i, ['%10s'%('%0.2f'%(j*amount)) for j in self.PlanParam[i]])

#============================================================================================================

class Contract():
    """Class Contract
       issue_dt - issue of contract
       duration - duration in months
    """

    dod_dic = {0: '0',
               1: '1+',
               2: '31+',
               3: '61+',
               4: '91+',
               5: 'WOF'
              }
    dod_cnt = 6 # кол-во состояний
    dod_states = np.eye(dod_cnt) # матрица состояний (для удобства использована единичная матрица)

    def __init__(self, cntr_id = 0, issue_dt = 0, duration = 0,
                 world = World, tariff = Tariff, amount = 100_000):
        self.cntr_id = cntr_id
        self.dod_id = 0        # начальное состояние контракта при выдачи: DOD = 0
        self.dod_state = self.dod_states[0] # np.array([1,0,0,0,0]) 
        self.dod_migration = world.dod_migration
        self.issue_dt = issue_dt
        self.mob = 0
        self.duration = duration
        self.closed_id = 0       # 0 - контратк открыт, 1 - закрыт
        self.wrtoff_id = 0       # 0 - контратк несписан, 1 - списан
        self.amount = amount
        self.tariff = tariff
        
    def next_month(self):
        if self.closed_id == 1:
            return None
           
        self.mob = self.mob + 1
        p = self.dod_migration[self.dod_id]                    # array of probabilities
        self.dod_id = np.random.choice(self.dod_cnt,1,p=p)[0]  # new state
        self.dod_state = self.dod_states[self.dod_id]

        if self.dod_id == 0 and self.mob >= self.duration: # погашение либо выздоровление с возвращением в график
            self.closed_id = 1
        
        if self.dod_id == 5 and self.mob >= self.duration + 12: # списание
            self.wrtoff_id = 1

        if self.wrtoff_id == 1 and self.mob >= self.duration + 24: # закрытие списанного контракта
            self.closed_id = 1
            
#============================================================================================================

class Portfolio():
    """Class Portfolio - Портфель - динамика 
        N - первая выдача при создании портфеля
        start_portfolio_dt - привязка портфеля к мировому времени - важно при наличии нескольких портфелей
    
    """
    def __init__(self, N = 10, duration = 36, start_portfolio_dt = 0, world = None, dwh = None):
        self.cntr_id = 0                                # счетчик контрактов
        self.start_portfolio_dt = start_portfolio_dt    # дата создания портфеля
        self.cntr_list = []                             # текущий сам портфель - список контрактов
        self.cntr_dic  = {}                             # справочник контрактов
        self.portfolio_age = 0                          # возрвст портфеля
        self.world = world                              # настройки внешнего мира
        self.dwh = dwh                                  # база данных

        # проведем первую выдачу - инициализация портфеля
#        self.issue(N, duration)
        # Заполним LI
#        self.fix_in_dwh()

    def issue(self, issue_plan = [(Tariff,0)]):
        for i in range(len(issue_plan)):
            self.cntr_id += 1
            cntr_tariff = issue_plan[i][0]
            cntr_amount = issue_plan[i][1]
            cntr = Contract(cntr_id = self.cntr_id,
                            issue_dt = self.portfolio_age,
                            duration = cntr_tariff.DUR,
                            world = self.world,
                            tariff = cntr_tariff,
                            amount = cntr_amount
                           )
            self.cntr_list.append(cntr)
            self.cntr_dic[self.cntr_id] = cntr

    def next_month(self, issue_plan = [(Tariff,0)], log = False):
        self.portfolio_age +=1

        # Для проверки - выведем все закрытые на этот момент контракты
        if log:
            test = [cntr.cntr_id for cntr in self.cntr_list if cntr.closed_id == 1 ]
            print('%04i' % self.portfolio_age, len(self.cntr_list), 'out ->',  test)            

        # Перезапишем список только открытыми контрактами 
        self.cntr_list = [cntr for cntr in self.cntr_list if cntr.closed_id == 0 ]
        
        # сдвинем существующий портфель, потом проведем выдачу новых         
        for cntr in self.cntr_list:
            cntr.next_month()
            
        # проведем выдачи
        self.issue(issue_plan)

        # Заполним LI
        self.fix_in_dwh()


    def fix_in_dwh_old(self): # Пример медленной вставки. Так не делать.
        ix = len(self.dwh.LI.index)
        for cnt in self.cntr_list:
            self.dwh.LI.loc[ix] = [cnt.cntr_id, self.portfolio_age, cnt.dod_id, cnt.mob]
            ix += 1

    def fix_in_dwh(self):
        fix_data = [[cnt.cntr_id, self.portfolio_age, cnt.dod_id, cnt.mob, cnt.wrtoff_id, cnt.closed_id] for cnt in self.cntr_list]
        self.dwh.LI = pd.concat([self.dwh.LI,
                                 pd.DataFrame(data=fix_data,
                                              columns=self.dwh.LI.columns)
                                ])
    def update_dwh_dic(self):
        update_data = [[cntr_id,
                        self.cntr_dic[cntr_id].issue_dt,
                        self.cntr_dic[cntr_id].wrtoff_id,
                        self.cntr_dic[cntr_id].closed_id,
                        self.cntr_dic[cntr_id].amount,
                        self.cntr_dic[cntr_id].duration,
                        self.cntr_dic[cntr_id].tariff.IR,
                        self.cntr_dic[cntr_id].tariff.name
                       ]
                       for cntr_id in self.cntr_dic]
        
        self.dwh.DMContract = pd.DataFrame(data=update_data,
                                           columns=self.dwh.DMContract.columns)


#============================================================================================================

#============================================================================================================

