import time
from autocfr.worker import Worker, VecWorker, GroupVecWorker
import numpy as np
import ray


class DiverContainer(Worker):
    @ray.remote
    def run(task):
        """执行除法。

        Args:
            task (dict）：执行除法所需信息
                a: 被除数
                b: 除数

        Return:
            result (dict): 任务执行结果[正常运行]
                state: succ
                worker_index: 任务执行器编号
                out: 除法结果
            result (dict): 任务执行结果[异常]
                state: succ
                worker_index: 任务执行器编号
                error: 异常对象
                info: 异常信息
        """
        a = task["a"]
        b = task["b"]
        result = {
            "worker_index": task["worker_index"],
            "group_index": task["group_index"],
            "a": a,
            "b": b,
        }
        try:
            time.sleep(a)
            out = a / b
        except Exception as e:
            result["status"] = "fail"
            result["error"] = e
            result["info"] = str(e)
        else:
            result["status"] = "succ"
            result["out"] = out
        return result


class Diver(Worker):
    def run(self, task):
        """执行除法。

        Args:
            task (dict）：执行除法所需信息
                a: 被除数
                b: 除数

        Return:
            result (dict): 任务执行结果[正常运行]
                state: succ
                worker_index: 任务执行器编号
                out: 除法结果
            result (dict): 任务执行结果[异常]
                state: succ
                worker_index: 任务执行器编号
                error: 异常对象
                info: 异常信息
        """
        try:
            result = self.get_result_dict(task)
            a = task["a"]
            b = task["b"]
            time.sleep(int(a))
            out = a / 0
            out = a / b
        except Exception as e:
            result["state"] = "fail"
            result["error"] = e
            result["info"] = str(e)
        else:
            result["state"] = "succ"
            result["out"] = out
        return result


def test_run():
    diver = Diver(1)
    result = diver.run(dict(a=1, b=0))
    assert result["state"] == "fail"


def atest_parallel_run():
    ray.init()
    vec_worker = VecWorker(3, Diver)
    for i in range(10):
        a = np.random.randint(low=0, high=100)
        b = np.random.randint(low=0, high=100)
        vec_worker.add_task(dict(a=a, b=b))
    for i in range(20):
        time.sleep(0.01)
        result = vec_worker.get_result()
        print(vec_worker.get_info())
    ray.shutdown()


def atest_parallel_run_sync():
    ray.init()
    vec_worker = VecWorker(2, Diver)
    tasks = []
    for i in range(10):
        a = np.random.randint(low=0, high=100)
        b = np.random.randint(low=0, high=100)
        tasks.append(dict(a=a, b=b))
    results = vec_worker.execute_tasks(tasks)
    for task, result in zip(tasks, results):
        print(task["a"], task["b"], task["a"] / task["b"], result["out"])
    ray.shutdown()


def atest_group_run():
    ray.init()
    group_vec_worker = GroupVecWorker(10, DiverContainer)
    # group_vec_worker.add_tasks([dict(a=1, b=2), dict(a=3, b=4)])
    group_vec_worker.add_tasks([dict(a=3, b=4), dict(a=3, b=7), dict(a=1, b=1)])
    group_vec_worker.add_tasks([dict(a=3, b=4), dict(a=5, b=0)])
    group_vec_worker.add_tasks([dict(a=1, b=0), dict(a=3, b=4), dict(a=3, b=0)])
    group_vec_worker.add_tasks([dict(a=1, b=4), dict(a=3, b=0), dict(a=2, b=4), ])
    # group_vec_worker.add_tasks([dict(a=1, b=1), dict(a=3, b=1)])

    for i in range(20):
        time.sleep(1)
        print(group_vec_worker.info())
        while True:
            result = group_vec_worker.get_result()
            if result is not None:
                print(result)
            else:
                break
    ray.shutdown()
