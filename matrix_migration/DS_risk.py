import numpy as np
import pandas as pd
from sklearn import metrics
from scipy.optimize import root_scalar              # Для создания модели
from sklearn.linear_model import LogisticRegression # Для калибровки модели


#============================================================================================================

class World():
    """Class World - Макромир, который задает начало отсчета времени,
        законы макроэкономики, ограничения регуляторов/ЦБ и остальное окружение.
        Так же здесь описывается поведение клиента, которое может зависеть от внешних факторов.

        dod_migration - матрицы вероятностей переходов по просрочкам (AVG-средняя, 0-контракты good, 1-контракты bad)
    """

    def __init__(self, seed = 0):
        self.seed = seed
        self.Time = 0
        self.Population_Mu = 0
        self.Population_Sigma = 1
        self.Fate_cutoff_pd = 0.3
        self.Fate_cutoff_score = np.log(self.Fate_cutoff_pd/(1-self.Fate_cutoff_pd)) # пересчитаем pd в score
        #                                  0    1+   31+   61+   91+   WOF
        self.dod_migration = {'AVG': # средняя по портфелю
                             np.array([[0.95, 0.05,    0,    0,    0,    0], #  0  (нет пропущенных платежей)
                                       [0.90, 0.05, 0.05,    0,    0,    0], #  1+ (1 пропущенный платеж)
                                       [0.10, 0.05, 0.05, 0.80,    0,    0], # 31+ (2 пропущенных платежа)
                                       [0.05, 0.05, 0.05, 0.05, 0.80,    0], # 61+ (3 пропущенных платежа)
                                       [0.01, 0.01, 0.02, 0.02, 0.04, 0.90], # 91+ (4 пропущенных платежа)
                                       [   0,    0,    0,    0,    0,    1]  # WOF (списание контракт)
                                      #[0.00, 0.00, 0.00, 0.00, 0.00, 1.00]  # TODO - добавить досрочное и частичнодочсрочное погашение кредита 
                                      ]),
                              0: # good - контракты, которые закрываются без списывания
                             np.array([[0.97, 0.03,    0,    0,    0,    0], #  0  (нет пропущенных платежей)
                                       [0.55, 0.32, 0.13,    0,    0,    0], #  1+ (1 пропущенный платеж)
                                       [0.28, 0.17, 0.12, 0.43,    0,    0], # 31+ (2 пропущенных платежа)
                                       [0.19, 0.06, 0.05, 0.11, 0.59,    0], # 61+ (3 пропущенных платежа)
                                       [0.10, 0.02, 0.01, 0.01, 0.86,    0], # 91+ (4 пропущенных платежа)
                                       [   0,    0,    0,    0,    0,    1]  # WOF (списание контракт)
                                      #[0.00, 0.00, 0.00, 0.00, 0.00, 1.00]  # TODO - добавить досрочное и частичнодочсрочное погашение кредита 
                                      ]),
                              1: # bad - контракты, которые закрываются списанием
                             np.array([[0.85, 0.15,    0,    0,    0,    0], #  0  (нет пропущенных платежей)
                                       [0.19, 0.22, 0.59,    0,    0,    0], #  1+ (1 пропущенный платеж)
                                       [0.04, 0.04, 0.05, 0.87,    0,    0], # 31+ (2 пропущенных платежа)
                                       [0.01, 0.01, 0.01, 0.03, 0.94,    0], # 61+ (3 пропущенных платежа)
                                       [0.01, 0.01, 0.01, 0.01, 0.93, 0.03], # 91+ (4 пропущенных платежа)
                                       [   0,    0,    0,    0,    0,    1]  # WOF (списание контракт)
                                      #[0.00, 0.00, 0.00, 0.00, 0.00, 1.00]  # TODO - добавить досрочное и частичнодочсрочное погашение кредита 
                                      ])
                             }
        for k in self.dod_migration:
            print('test(%s) - ' % k, [i.sum() for i in self.dod_migration[k]])
            assert self.dod_migration[k].shape[0] == self.dod_migration[k].shape[1], 'проверка на квадратность - key=%s' % k
            assert [i.sum() for i in self.dod_migration[k]] == [1 for i in range(len(self.dod_migration[k]))], '1 in sum of row - key=%s' % k

        print('Hello World!')

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
        fate  = [1 if scr > self.Fate_cutoff_score else 0 for scr in score]
        self.seed += 1
        return [score, fate]

#============================================================================================================

class Model():
    """Class Model - рисковые модели
    """

    id = 0
    model_dic = {}

    def __init__(self, model_mu=0, model_sigma=1):
        Model.id += 1
        Model.model_dic[Model.id] = self
        self.Model_id = Model.id
        self.Model_Mu = model_mu
        self.Model_Sigma = model_sigma
        self.Calib_koef = (0, 1)          # Калибровка по умолчанию
        self.log_dic = {}                 # Для неструктурированных записей

    # Краткая инфа о модели
    def info(self):
        print(self.Model_id, self.Model_Mu, self.Model_Sigma)

    def Score(self, factors):
        model_diff = np.random.normal(self.Model_Mu, self.Model_Sigma, len(factors))
#        print(model_diff)
#        print(factors)
        # Вернем смещенный скор
        return factors + model_diff

    # Применение калибровки
    def PD(self, score):
        A = self.Calib_koef[0]
        B = self.Calib_koef[1]
        return 1/(1+np.exp(-A-B*score))

#============================================================================================================

class Bank_DS():
    """Здесь будут жить фишки отдела DS
    """

    def __init__(self):
        self.N_grade = 16   # Кол-во рейтингов по умолчанию.
        print('Hello DS!')


    def sigmoid(self, x):
        '''Функция сигмоиды'''
        return 1/(1+np.exp(-x))


    def logit(self, p):
        '''Функция обратная сигмоиде'''
        return np.log(p / (1 - p))

    # Расчет Gini 
    def gini(self, df, target='BADFLAG', score='SCORE', plot=False):
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

    # Функция границы мастер шкалы
    def func_of_threshold(self, i):
        return 2**((-self.N_grade+i)/2)
   
    # Расчет рейтинга. Можно было бы вывести обратную функцию от границы, но, возможно, это не лучшее решение.
    def grade(self, pd):
        for i in range(1,17):
            R_pd = self.func_of_threshold(i)
            if pd < R_pd:
                return i

    # Распечатать границы мастер шкалы
    def print_master_scale(self):
        print('Grage Right_value')
        for g in range(1, self.N_grade+1):
            print('%5s %0.3g'%(g, self.func_of_threshold(g)))


    def create_model(self, gini = None, tto_sample = None, N_range = 100):
        ''' Создание модели.
            Модель создается на выданных кредитах подбором значений зашумления предопределенного скора.
            Здесь подбираются шум таким образом, чтобы качество модели (gini) соответствовало заданому значению.
        '''
        if not gini:
            return Model()

        y_gini = gini
        model = Model()
        DF = tto_sample.copy()
        dic_fun = {}

        def f_gini(s):
            s = round(s,3)
            model.Model_Sigma = s

            if s == 0:
                y = self.gini(DF, target = 'BADFLAG', score = 'FATED_SCORE')
                dic_fun[s] = y
                return  y - y_gini

            test = []
            for i in range(N_range):
                DF['M1'] = model.Score(np.array(DF['FATED_SCORE'].values))
                test.append(self.gini(DF, target = 'BADFLAG', score = 'M1'))
            y = np.array(test).mean()
            dic_fun[s] = y

            return y - y_gini

        # Поиск корня в границах bracket.
        sol = root_scalar(f_gini, bracket=[0, 5], xtol = 0.01, maxiter = 10)

        model.log_dic['root_scalar'] = sol
        model.log_dic['dic_fun'] = dic_fun

        return model


    def calibration(self, model, sample, CT = None):
        ''' Калибровка модели за заданном семпле
            model  - калибруемая модель
            sample - семпел калибровки должен содержать поля MODEL_SCORE и BADFLAG
            CT     - центральная тенденция. Если CT заполнено, то проводится калибровка на уровень CT.
        '''

        if CT:
            DR = sample['BADFLAG'].mean()
            weight_defl = CT/DR
            weight_good = (1-CT)/(1-DR)
            sample_weight = [weight_defl if target == 1 else weight_good for target in sample['BADFLAG']]
            model.log_dic['weight_defl'] = weight_defl
            model.log_dic['weight_good'] = weight_good
            model.log_dic['CT'] = CT
            model.log_dic['DR'] = DR
        else:
            sample_weight = None

        LR = LogisticRegression(random_state=42)
        LR.fit(X = sample[['MODEL_SCORE']], y = sample['BADFLAG'], sample_weight = sample_weight)

        model.log_dic['LR'] = LR
        model.Calib_koef = (LR.intercept_[0], LR.coef_[0][0])

        return model.Calib_koef




#============================================================================================================

class DWH_DB():
    """Class DWH - база данных
    """
    def __init__(self):
        self.LI = pd.DataFrame(columns = ['PORTFOLIO_ID',
                                          'CNTR_ID',
                                          'SD',
                                          'DOD_ID',
                                          'MOB',
                                          'WRTOFF_ID',
                                          'CLOSED_ID'
                                          ])
        self.DMContract = pd.DataFrame(columns = ['PORTFOLIO_ID',
                                                  'CNTR_ID',
                                                  'ISSUE_DT',
                                                  'WRTOFF_DT',
                                                  'CLOSED_DT',
                                                  'AMOUNT',
                                                  'DURATION',
                                                  'IR',
                                                  'TARIFF',
                                                  'MODEL_SCORE',
                                                  'FATED_SCORE',
                                                  'FATED_RESULT',
                                                  'MODEL_ID',
                                                  'NUM_IN_QUEUE',
                                                 ])

    def update_dwh_dic(self):
        update_data = [[Contract.cntr_dic[cntr_id].portfolio_id,
                        cntr_id,
                        Contract.cntr_dic[cntr_id].issue_dt,
                        Contract.cntr_dic[cntr_id].wrtoff_id,
                        Contract.cntr_dic[cntr_id].closed_id,
                        Contract.cntr_dic[cntr_id].amount,
                        Contract.cntr_dic[cntr_id].duration,
                        Contract.cntr_dic[cntr_id].tariff.IR,
                        Contract.cntr_dic[cntr_id].tariff.name,
                        Contract.cntr_dic[cntr_id].model_score,
                        Contract.cntr_dic[cntr_id].fated_score,
                        Contract.cntr_dic[cntr_id].fated_result,
                        Contract.cntr_dic[cntr_id].model_id,
                        Contract.cntr_dic[cntr_id].number_in_queue,
                       ]
                       for cntr_id in Contract.cntr_dic]
        
        self.DMContract = pd.DataFrame(data=update_data,
                                           columns=self.DMContract.columns)

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
    dod_cnt = 6                  # кол-во состояний
    dod_states = np.eye(dod_cnt) # матрица состояний (для удобства использована единичная матрица)
    id = 0                       # сквозной номер
    cntr_dic  = {}               # справочник контрактов

    def __init__(self, portfolio_id, issue_dt = 0, duration = 0,
                 world = World, tariff = Tariff, amount = 100_000,
                 model_score = None,
                 fated_score = None,
                 fated_result = None,
                 model_id = None,
                 number_in_queue = None
                ):
        Contract.id += 1
        self.cntr_id = Contract.id
        self.portfolio_id = portfolio_id
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
        self.model_score = model_score
        self.fated_score = fated_score
        self.fated_result = fated_result
        self.model_id = model_id
        self.number_in_queue = number_in_queue
        self.p = 0
        Contract.cntr_dic[Contract.id] = self    # сквозной справочник контрактов

    def next_month(self, dod_migration_type=None):
        if self.closed_id == 1:
            return None

        if not dod_migration_type:
            dod_migration_type = self.fated_result

        self.mob = self.mob + 1
        self.p = self.dod_migration[dod_migration_type][self.dod_id]   # array of probabilities
        self.dod_id = np.random.choice(self.dod_cnt, 1, p = self.p)[0]     # new state
        self.dod_state = self.dod_states[self.dod_id]

        if self.dod_id == 0 and self.mob >= self.duration: # погашение либо выздоровление с возвращением в график
            self.closed_id = 1
        
        if self.wrtoff_id == 0 and self.dod_id == 5 and self.mob >= self.duration + 12: # списание
            self.wrtoff_id = 1

        if self.wrtoff_id == 1 and self.mob >= self.duration + 24: # закрытие списанного контракта
            self.closed_id = 1
            
#============================================================================================================

class Portfolio():
    """Class Portfolio - Портфель - динамика 
        N - первая выдача при создании портфеля
        start_portfolio_dt - привязка портфеля к мировому времени - важно при наличии нескольких портфелей
    
    """

    id = 0
    portfolio_dic = {}

    def __init__(self, N = 10, start_portfolio_dt = 0, world = None, dwh = None):
        Portfolio.id += 1
        self.portfolio_id = Portfolio.id                 # id портфеля
        self.start_portfolio_dt = start_portfolio_dt    # дата создания портфеля
        self.cntr_list = []                             # текущий сам портфель - список контрактов
        self.portfolio_age = 0                          # возраст портфеля
        self.world = world                              # настройки внешнего мира
        self.dwh = dwh                                  # база данных
        Portfolio.portfolio_dic[Portfolio.id] = self    # сквозной справочник портфелей

        # проведем первую выдачу - инициализация портфеля
#        self.issue(N, duration)
        # Заполним LI
#        self.fix_in_dwh()

    def issue(self, issue_plan = [(Tariff,0)], pd_cutoff = 0.1, model = None):
        score_cutoff = np.log(pd_cutoff / (1 - pd_cutoff))
        approved_list = []
        N_flow = len(issue_plan)
        i = 0

        # Есть план выдач. Впускаем по плану и часть одобряем.
        # Потом опять впускаем и одобряем, пока не одобрим плановое число (или больше).
        while len(approved_list) < N_flow:
            M_flow = N_flow * 10  # впускаем больше плана
            score_list, res_list = self.world.get_god_score(M_flow)
            model_score_list = model.Score(score_list)
            approved_list.extend([(ms, fs, fr, i) for ms, fs, fr, i in zip(model_score_list,
                                                                           score_list,res_list,
                                                                           range(i*M_flow,(i+1)*M_flow)
                                                                          ) if ms < score_cutoff])
            i+=1
#            print('--')
#            print(len(approved_list))

#        print('----')
#        print(score_list)
#        print(approved_list)
        i_queue_last = 0
        for (cntr_tariff, cntr_amount), (ms, fs, fr, i_queue) in zip(issue_plan,approved_list):
#            print(cntr_tariff, cntr_amount, ms, fs, fr)
            cntr = Contract(portfolio_id = self.portfolio_id,
                            issue_dt = self.portfolio_age,
                            duration = cntr_tariff.DUR,
                            world = self.world,
                            tariff = cntr_tariff,
                            amount = cntr_amount,
                            model_score = ms,
                            fated_score = fs,
                            fated_result = fr,
                            model_id = model.Model_id,
                            number_in_queue = i_queue - i_queue_last,
                           )
            self.cntr_list.append(cntr)

            # Обновим последний номер очереди.
            i_queue_last = i_queue

    def next_month(self, issue_plan = [(Tariff,0)], log = False, dod_migration_type = None, model = None, pd_cutoff = None):
        self.portfolio_age +=1

        # Для проверки - выведем все закрытые на этот момент контракты
        if log:
            test = [cntr.cntr_id for cntr in self.cntr_list if cntr.closed_id == 1 ]
            print('%04i' % self.portfolio_age, len(self.cntr_list), 'out ->',  test)            

        # Перезапишем список только открытыми контрактами 
        self.cntr_list = [cntr for cntr in self.cntr_list if cntr.closed_id == 0 ]
        
        # сдвинем существующий портфель, потом проведем выдачу новых         
        for cntr in self.cntr_list:
            cntr.next_month(dod_migration_type = dod_migration_type)
            
        # проведем выдачи
        self.issue(issue_plan = issue_plan, pd_cutoff = pd_cutoff, model = model)

        # Заполним LI
        self.fix_in_dwh()


    def fix_in_dwh_old(self): # Пример ме дленной вставки. Так не делать.
        ix = len(self.dwh.LI.index)
        for cnt in self.cntr_list:
            self.dwh.LI.loc[ix] = [cnt.cntr_id, self.portfolio_age, cnt.dod_id, cnt.mob]
            ix += 1

    def fix_in_dwh(self):
        fix_data = [[self.portfolio_id, cnt.cntr_id, self.portfolio_age, cnt.dod_id, cnt.mob, cnt.wrtoff_id, cnt.closed_id] for cnt in self.cntr_list]
        self.dwh.LI = pd.concat([self.dwh.LI,
                                 pd.DataFrame(data=fix_data,
                                              columns=self.dwh.LI.columns)
                                ])

    # Краткая инфа о портфеле
    def info(self):
        print('ID = %i' % self.portfolio_id, 
              'Возраст портеля %i мес' % self.portfolio_age,
              'Кол-во контрактов %i' % len(self.cntr_list),
             )


#============================================================================================================
#============================================================================================================

