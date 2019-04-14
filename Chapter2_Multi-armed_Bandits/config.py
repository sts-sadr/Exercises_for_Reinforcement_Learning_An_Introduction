# デフォルトのハイパーパラメータ. 最もよくある値を入れているだけで最適な値とは限らない

# problem config
PROBLEM_NAME = 'stationary'  # 扱うBandit問題の種類. 'stationary' OR 'nonstationary' OR 'contextual'

# 共通パラメータ
K_ARMS = 10  # 行動の種類数. 任意の整数 >= 1
REWARD_STD = 1.0  # 報酬の標準偏差

# 固有パラメータ
N_STATES = 2  # for contextual. 状態数. 任意の整数 >= 1. contextual以外ではここの値に関わらず常に1
PROBLEM_STD = 1.0  # for stationary, contextual. 真の価値の標準偏差
CHANGE_STD = 0.01  # for nonstationary. 加算される毎ステップの価値の変化の標準偏差
STATE_INFO = True  # for contextual. 状態情報を用いるかどうか


# solver config
SOLVER_NAME = 'epsilon-greedy'  # 使うアルゴリズム. 'epsilon-greedy' OR 'UCB1' OR 'policygradient'

# 共通パラメータ
STEP_SIZE = 0.1  # 学習率. -1のときは(1/ステップ数)を使う（ようにsolverを書いてある）

# 固有パラメータ
INITIAL_VALUE = 0.0  # for epsilon-greedy, UCB1. 価値推定時の初期値
EPSILON = 0.1  # for epsilon-greedy. 探索率
CONF_COEFF = 2  # for UCB1. confidence項にかかる係数


# other config
N_RUNS = 20
N_STEPS = 10000
