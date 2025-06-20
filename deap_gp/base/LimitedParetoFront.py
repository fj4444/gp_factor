from copy import deepcopy
from deap import tools

# !!这个是自己定义的类，对DEAP里的帕累托前沿的包装，原生的pareto front只能维护第一前沿，这里的办法就是每次维护第一前沿，如果数量不够就把剩下的个体归拢起来再继续取第一前沿

class LimitedParetoFront():
    """可以指定包含个体数量的ParetoFront Hall of fame, 其中个体可能来自任何一个前沿面,
    只保证每次更新后找到的个体, 不被 新的population个体和旧的hof个体组成的集合 的剩余个体支配,
    如果maxsize个个体中有些和未被选入的个体互不支配, 则被选入的个体的拥挤距离比未被选入的更小
    """

    def __init__(self, maxsize, similar=eq):
        self.pareto_front = tools.ParetoFront()
        self.maxsize = maxsize
        self.keys = list()
        self.items = list()
        self.similar = similar

    def update(self, population):
        assert len(population)>=self.maxsize, "设置的优胜种群大小不应超过整体种群"
        pop = deepcopy(population)
        pop.extend(self)
        self.clear()
        self.pareto_front.update(pop)
        if len(self) + len(self.pareto_front) > self.maxsize:
            # 如果超出大小限制，根据拥挤距离裁剪
            crowding_dist = tools.emo.assignCrowdingDist(self.pareto_front)
            items = [(ind, dist, fitness) for ind, dist, fitness in zip(self.pareto_front, crowding_dist, self.pareto_front.keys)]
            # 按拥挤距离降序排序
            items.sort(key=lambda x: x[1], reverse=True)
            # 保留拥挤距离最大的maxsize个个体
            self.items = [item[0] for item in items[:self.maxsize - len(self)]]
            self.keys = [item[2] for item in items[:self.maxsize - len(self)]]

        else:
            self.items.extend(self.pareto_front.items)
            self.keys.extend(self.pareto_front.keys)
            if len(self) < self.maxsize:
                ind_to_remove = set(self.pareto_front)
                pop = [ind for ind in pop if ind not in ind_to_remove]
                self.pareto_front.clear()
                self.update(pop)


    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        return reversed(self.items)

    def __str__(self):
        return str(self.items)