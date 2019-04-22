# デフォルトのハイパーパラメータ. 最もよくある値を入れているだけで最適な値とは限らない


# ------------------------
#      problem config
# ------------------------
PROBLEM_NAME = 'stationary'  # 扱うBandit問題の種類. 'stationary' OR 'nonstationary' OR 'contextual'

# 共通パラメータ
K_ARMS = 10  # 行動の種類数. 任意の整数 >= 1
REWARD_STD = 1.0  # 報酬の標準偏差

# 固有パラメータ
PROBLEM_MEAN = 0  # for stationary, contextual. 真の価値の平均
PROBLEM_STD = 1.0  # for stationary, contextual. 真の価値の標準偏差
CHANGE_STD = 0.01  # for nonstationary. 加算される毎ステップの価値の変化の標準偏差
N_STATES = 2  # for contextual. 状態数. 任意の整数 >= 1. contextual以外ではこの値に関わらず状態数は常に1
STATE_INFO = True  # for contextual. 状態情報を返すかどうか


# -----------------------
#      solver config
# -----------------------
SOLVER_NAME = 'epsilon-greedy'  # 使うアルゴリズム. 'epsilon-greedy' OR 'UCB1' OR 'policygradient'

# 共通パラメータ
STEP_SIZE = 0.1  # 学習率. -1のときは(1/ステップ数)を使う（ようにsolverを書いてある）

# 固有パラメータ
INITIAL_VALUE = 0.0  # for epsilon-greedy, UCB1. 価値推定時の初期値
EPSILON = 0.1  # for epsilon-greedy. 探索率
CONF_COEFF = 1  # for UCB1. confidence項にかかる係数
BASELINE_STEP_SIZE = -1  # for policygradient. ベースライン更新のステップサイズ. -1でサンプル平均
WITH_BASELINE = True  # for policygradient. ベースラインを使うかどうか


# ----------------------
#      other config
# ----------------------
N_RUNS = 50  # エピソードを回す回数
N_STEPS = 1000  # 1エピソードのステップ数
