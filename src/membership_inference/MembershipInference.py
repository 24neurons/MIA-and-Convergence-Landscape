from .ShadowModel import ShadowModel
from .AttackModel import AttackModel
"""
    Membership Inference Attack
    This is the membership inference attack implementation
    based on the shadow model architecture, which is described in the paper
    https://arxiv.org/abs/1610.05820
"""


class MembershipInference:
    """
        The combination of target model, shadow models and attack models
    """
    def __init__(self, target_model, shadow_models, shadow_train_size,
                 attack_base_model, randomSeed):
        self.shadow_models = shadow_models
        self.shadow_train_size = shadow_train_size
        self.attack_base_model = attack_base_model

        self.tm = target_model
        self.sm = None
        self.am = None
        self.number_of_classes = None
        self.shadow_results = []
        self.seed = randomSeed

    def trainShadow(self, X, y, numIter):
        self.sm = ShadowModel(listOfModels=self.shadow_models,
                              trainDataSize=self.shadow_train_size,
                              randomSeed=self.seed)

        self.shadow_results = self.sm.fitTransform(X, y, numIter)
        self.number_of_classes = len(y[0])

    def trainAttack(self, numIter):

        self.am = AttackModel(baseModel=self.attack_base_model)
        self.am.fit(self.shadow_results, self.number_of_classes, numIter)

    def attack(self, X, y, prob=False):

        target_pred = self.tm(X)

        if prob:
            return self.am.predict_membership_prob(target_pred=target_pred,
                                                   true_class=y)
        else:
            return self.am.predict_membership_status(target_pred=target_pred,
                                                     true_class=y)
