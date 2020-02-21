from pyomo.environ import *

# infinity = float('inf')

model = AbstractModel()

# Create Parameter for Number of Products
model.N = Param(within=PositiveIntegers)

model.I = RangeSet(1, model.N)

# Create Parameter for V's
model.v0 = Param(within=Reals)
model.v1 = Param(model.I, within=Reals, default=0.)
model.v2 = Param(model.I, model.I, within=Reals, default=0)
model.v3 = Param(model.I, model.I, model.I, within=Reals, default=0)

# Create Parameter for Prices of Products

model.r = Param(model.I, within=PositiveReals)

# Create Variable X's for selecting a product
model.x = Var(model.I, within=Binary)


def objective_rule(model):
    # n=model.N
    objective_expr_num = sum((model.v1[i] * model.r[i] * model.x[i] for i in model.I))
    # print("Step1 NUM:",objective_expr_num)
    objective_expr_num += 0.5 * sum(
        (model.v2[i, j] * (model.r[i] + model.r[j]) * model.x[i] * model.x[j] for i in model.I for j in model.I if
         (not (i == j))))
    # print("Step2 NUM:",objective_expr_num)
    objective_expr_num += (1 / 6) * sum(
        (model.v3[i, j, k] * (model.r[i] + model.r[j] + model.r[k]) * model.x[i] * model.x[j] * model.x[k] for i in
         model.I for j in model.I for k in model.I if ((not i == j) & (not j == k) & (not k == i))))
    # print("Step3 NUM:",objective_expr_num)
    objective_expr_den = model.v0
    objective_expr_den += sum((model.v1[i] * model.x[i] for i in model.I))
    # print("Step1 Den:",objective_expr_den)
    objective_expr_den += 0.5 * sum(
        (model.v2[i, j] * model.x[i] * model.x[j] for i in model.I for j in model.I if (not (i == j))))
    # print("Step2 Den:",objective_expr_den)
    objective_expr_den += (1 / 6) * sum(
        (model.v3[i, j, k] * model.x[i] * model.x[j] * model.x[k] for i in model.I for j in model.I for k in model.I if
         ((not i == j) & (not j == k) & (not k == i))))
    # print("Step3 Den:",objective_expr_den)
    return objective_expr_num / objective_expr_den


model.revenue = Objective(rule=objective_rule, sense=maximize)
