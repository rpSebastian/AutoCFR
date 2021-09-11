import threading
import time
import ray
from queue import Queue


class Worker:
    def __init__(self, index):
        self.index = index

    def run(self, task):
        """Worker抽象执行方法，输入task，返回result。

        Args:
            task (dict): 任务执行所需信息

        Return:
            results (dict): 任务执行结果[正常运行]
                state: succ
                worker_index: 任务执行器编号
                group_index (optional): 任务所在组编号
            results (dict): 任务执行结果[异常]
                state: fail
                worker_index: 任务执行器编号
                group_index (optional): 任务所在组编号
                error: 异常对象
                info: 异常信息

        """
        return NotImplemented

    def get_result_dict(self, task):
        result = {"worker_index": self.index}
        if "group_index" in task:
            result["group_index"] = task["group_index"]
        return result

    @classmethod
    def as_remote(
        cls,
        num_cpus: int = 1,
        num_gpus: int = 0,
        memory: int = None,
        object_store_memory: int = None,
        resources: dict = None,
    ) -> type:
        """[summary]

        Args:
            num_cpus (int, optional): [description]. Defaults to 1.
            num_gpus (int, optional): [description]. Defaults to 0.
            memory (int, optional): [description]. Defaults to None.
            object_store_memory (int, optional): [description]. Defaults to None.
            resources (dict, optional): [description]. Defaults to None.

        Returns:
            type: [description]
        """
        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)


class Counter:
    def __init__(self):
        self.waiting = 0
        self.running = 0
        self.finished = 0
        self.removed = 0
        self.error = 0
        self.outputed = 0
        self.finished_removed = 0
        self.lock = threading.Lock()

    def cum_waiting(self):
        with self.lock:
            self.waiting += 1

    def waiting_to_removed(self):
        self.waiting -= 1
        self.removed += 1

    def running_to_removed(self):
        self.running -= 1
        self.removed += 1

    def waiting_to_running(self):
        with self.lock:
            self.waiting -= 1
            self.running += 1

    def running_to_error(self, num=1):
        with self.lock:
            self.running -= num
            self.error += num

    def running_to_finished(self, num=1):
        with self.lock:
            self.running -= num
            self.finished += num

    def finished_to_finished_removed(self, num=1):
        with self.lock:
            self.finished -= num
            self.finished_removed += num

    def remove_finished(self, num=1):
        with self.lock:
            self.finished -= num
            self.outputed += num

    def state(self):
        with self.lock:
            state = {
                "waiting": self.waiting,
                "running": self.running,
                "finished": self.finished,
                "removed": self.removed,
                "finished_removed": self.finished_removed,
                "error": self.error,
                "outputed": self.outputed,
            }
        return state

    def info(self):
        state = self.state()
        info = ", ".join([k + ": " + str(v) for k, v in state.items()])
        return info


class VecWorker:
    def __init__(self, num_workers, worker_cls, **worker_kwargs):
        self.num_workers = num_workers
        self.worker_cls = worker_cls
        self.workers = {}
        self.worker_ready = {}
        self.worker_kwargs = worker_kwargs
        self.add_workers()
        self.task_waiting_queue = Queue()
        self.task_finished_queue = Queue()
        self.results_ref = []
        self.start_working()
        self.counter = Counter()

    def add_workers(self):
        for work_index in range(self.num_workers):
            self.add_worker(work_index)

    def add_worker(self, worker_index):
        remote_worker_cls = self.worker_cls.as_remote().remote
        remote_worker = remote_worker_cls(index=worker_index, **self.worker_kwargs)
        self.workers[worker_index] = remote_worker
        self.worker_ready[worker_index] = True

    def add_task(self, task):
        self.task_waiting_queue.put(task)
        self.counter.cum_waiting()

    def start_working(self):
        self.t = threading.Thread(target=self.keep_working)
        self.t.setDaemon(True)
        self.t.start()

    def keep_working(self):
        while True:
            time.sleep(0.01)
            ready_indexs = [k for k, v in self.worker_ready.items() if v]
            # 循环分配任务给所有空闲的执行器
            for worker_index in ready_indexs:
                if not self.task_waiting_queue.empty():
                    task = self.task_waiting_queue.get()
                    self.worker_ready[worker_index] = False
                    result_ref = self.workers[worker_index].run.remote(task)
                    self.results_ref.append(result_ref)
                    self.counter.waiting_to_running()

            # 循环获取所有已执行完成的结果
            while len(self.results_ref) > 0:
                # 非阻塞获取一个评估结果
                result_ref, self.results_ref = ray.wait(self.results_ref, timeout=0.01)
                # 获取得到结果
                if len(result_ref) == 1:
                    result = ray.get(result_ref[0])
                    worker_index = result["worker_index"]
                    del result["worker_index"]
                    self.worker_ready[worker_index] = True
                    self.task_finished_queue.put(result)
                    self.counter.running_to_finished()
                else:
                    # 没有已执行完成的结果，退出循环
                    break

    def get_result(self):
        """检查结果队列，若队列中有任务执行的结果，返回执行结果，否则返回None

        Returns:
            result (None): 评估未完成
            result (dict): 结果保存字典[正常运行]
                state: succ
            result (dict): 结果保存字典[异常]
                state: fail
                error: 异常对象
                info: 异常信息
        """
        result = None
        if not self.task_finished_queue.empty():
            result = self.task_finished_queue.get()
            self.counter.remove_finished()
        return result

    def execute_tasks(self, tasks):
        """同步执行所有任务，返回每个任务的执行结果

        Returns:
            results (list): 每个任务对应的结果
                result (dict): 结果保存字典[正常运行]
                    state: succ
                result (dict): 结果保存字典[异常]
                    state: fail
                    error: 异常对象
                    info: 异常信息
        """
        results = []
        num_tasks = len(tasks)
        for turn_index in range(num_tasks // self.num_workers):
            results_ref = []
            for worker_index in range(self.num_workers):
                task_index = turn_index * self.num_workers + worker_index
                task = tasks[task_index]
                result_ref = self.workers[worker_index].run.remote(task)
                results_ref.append(result_ref)
            results.extend(ray.get(results_ref))

        turn_index = num_tasks // self.num_workers
        results_ref = []
        for worker_index in range(num_tasks % self.num_workers):
            task_index = turn_index * self.num_workers + worker_index
            task = tasks[task_index]
            result_ref = self.workers[worker_index].run.remote(task)
            results_ref.append(result_ref)
        results.extend(ray.get(results_ref))
        return results

    def state(self):
        return self.counter.state()

    def info(self):
        return self.counter.info()

    @classmethod
    def as_remote(
        cls,
        num_cpus: int = 1,
        num_gpus: int = 0,
        memory: int = None,
        object_store_memory: int = None,
        resources: dict = None,
    ) -> type:
        """[summary]

        Args:
            num_cpus (int, optional): [description]. Defaults to 1.
            num_gpus (int, optional): [description]. Defaults to 0.
            memory (int, optional): [description]. Defaults to None.
            object_store_memory (int, optional): [description]. Defaults to None.
            resources (dict, optional): [description]. Defaults to None.

        Returns:
            type: [description]
        """
        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)


class Group:
    total_index = 0

    def __init__(self, tasks):
        self.index = Group.total_index
        Group.total_index += 1
        self.tasks = tasks
        self.results = []
        self.count = len(tasks)


class GroupVecWorker(VecWorker):
    """以组为单位评估任务，当一个任务出现异常停止其他任务。

    由于ray.cancel不适用于acotr，因此修改为调用远程函数的方式进行评估。
    """

    def __init__(self, num_workers, worker_cls, **worker_kwargs):
        super().__init__(num_workers, worker_cls, **worker_kwargs)
        self.total_task_index = 0
        self.groups = {}

    def add_worker(self, worker_index):
        worker = self.worker_cls(index=worker_index)
        self.workers[worker_index] = worker
        self.worker_ready[worker_index] = True

    def add_tasks(self, tasks):
        group = Group(tasks)
        for task in tasks:
            task["group_index"] = group.index
            task["status"] = "alive"
            self.add_task(task)
        self.groups[group.index] = group

    def get_result(self):
        """检查结果队列，若队列中有任务执行的结果，返回执行结果，否则返回None

        Returns:
            result (None): 评估未完成
            result (dict): 结果保存字典[正常运行]
                state: succ
            result (dict): 结果保存字典[异常]
                state: fail
                error: 异常对象
                info: 异常信息
        """
        result = None
        if not self.task_finished_queue.empty():
            result = self.task_finished_queue.get()
            if result["status"] == "succ":
                self.counter.remove_finished(len(result["results"]))
        return result

    def keep_working(self):
        while True:
            time.sleep(0.01)
            ready_indexs = [k for k, v in self.worker_ready.items() if v]
            # 循环分配任务给所有空闲的执行器
            for worker_index in ready_indexs:
                if not self.task_waiting_queue.empty():
                    task = self.task_waiting_queue.get()
                    task["worker_index"] = worker_index
                    if task["status"] == "killed":
                        continue
                    self.worker_ready[worker_index] = False
                    result_ref = self.workers[worker_index].run.remote(
                        task, **self.worker_kwargs
                    )
                    task["result_ref"] = result_ref
                    task["worker_index"] = worker_index
                    self.results_ref.append(result_ref)
                    self.counter.waiting_to_running()

            # 循环获取所有已执行完成的结果
            while len(self.results_ref) > 0:
                # 非阻塞获取一个评估结果
                result_ref, self.results_ref = ray.wait(self.results_ref, timeout=0.01)
                # 获取得到结果
                if len(result_ref) == 1:
                    result = ray.get(result_ref[0])
                    worker_index = result["worker_index"]
                    group_index = result["group_index"]
                    del result["worker_index"]
                    del result["group_index"]
                    self.worker_ready[worker_index] = True

                    group = self.groups[group_index]
                    if result["status"] == "succ":
                        group.results.append(result)
                        self.counter.running_to_finished()
                        if len(group.results) == group.count:
                            self.task_finished_queue.put(
                                dict(status="succ", results=group.results)
                            )

                    # 标记执行任务
                    for group_task in group.tasks:
                        if (
                            "result_ref" in group_task
                            and group_task["result_ref"] == result_ref[0]
                        ):
                            group_task["status"] = "evaluated"

                    # 如果一个任务执行出现异常，终止其他正在执行的任务或者将任务从等待队列中移除
                    if result["status"] == "fail":
                        self.counter.running_to_error()
                        for group_task in group.tasks:
                            # 正在执行的任务
                            if "result_ref" in group_task:
                                if group_task["status"] != "alive":
                                    continue
                                group_task_result_ref = group_task["result_ref"]
                                group_task_worker_index = group_task["worker_index"]
                                ray.cancel(group_task_result_ref)
                                self.results_ref.remove(group_task_result_ref)
                                self.worker_ready[group_task_worker_index] = True
                                self.counter.running_to_removed()
                            # 等待中的任务
                            else:
                                group_task["status"] = "killed"
                                self.counter.waiting_to_removed()
                        self.task_finished_queue.put(
                            dict(status="fail", results=[result])
                        )
                        self.counter.finished_to_finished_removed(len(group.results))
                else:
                    # 没有已执行完成的结果，退出循环
                    break
