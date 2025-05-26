class LimitedParetoFront(tools.ParetoFront):
    
    def __init__(self, maxsize, similar=None):
        super().__init__(similar)
        self.maxsize = maxsize
    
    def update(self, population):
        """更新Pareto前沿，并在必要时裁剪大小"""
        # 首先使用标准ParetoFront更新
        super().update(population)
        
        # 如果超出大小限制，根据拥挤距离裁剪
        if len(self) > self.maxsize:
            crowding_dist = tools.emo.assignCrowdingDist(self)
            items = [(ind, dist) for ind, dist in zip(self, crowding_dist)]
            # 按拥挤距离降序排序
            items.sort(key=lambda x: x[1], reverse=True)
            # 保留拥挤距离最大的maxsize个个体
            self.items = [item[0] for item in items[:self.maxsize]]

hof = LimitedParetoFront(maxsize=50)
