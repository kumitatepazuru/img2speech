import time
from multiprocessing import Pool


def nijou(inputs):
    x = inputs
    print('input: %d' % x)
    time.sleep(10)
    retValue = x * x
    print('double: %d' % retValue)
    return retValue


if __name__ == "__main__":

    # Pool()を定義
    p = Pool()

    # プロセスを2つ非同期で実行
    result = p.apply_async(nijou, args=[3])
    result2 = p.apply_async(nijou, args=[5])

    # 1秒間隔で終了チェックして終了したら結果を表示
    result.ready()
    print(result.get())
    print(result2.get())

    p.close()
