library(ISLR)
library(MASS)
names(Boston)

attach(Boston)

# Simple Linear Regression For Boston Dataset --> response : medv && predictor : lstat
linear_model = lm(medv ~ lstat) # lm(y ~ x) --> y is response and x is predictor
linear_model
summary(linear_model)
names(linear_model)

confint(linear_model) # get confindence intervals for coefficients

predict(linear_model, data.frame(lstat = c(5, 10, 15)), interval = "confidence") # predict with confidence intervals for x
predict(linear_model, data.frame(lstat = c(5, 10, 15)), interval = "prediction") # predict with confidence intervals for y

plot(lstat, medv, col = "red", pch = 20) 
abline(linear_model, lwd = 3, col = "blue") # draw predicted line for data
  

par(mfrow = c(2, 2)) # divide screen in 2 * 2 grid of panels
plot(linear_model, pch = 20, col = "blue")

plot(predict(linear_model), residuals(linear_model), pch = 20, col = "blue")
plot(predict(linear_model), rstudent(linear_model), pch = 20, col = "red")


# Multiple Linear Regression --> syntax : lm(y ~ x1+x2+x3+...)
multiple_linear_model = lm(medv~lstat+age, data = Boston)
plot(age, medv, pch = 20, col = "red")

multiple_linear_model = lm(medv~., data = Boston) # predictors are all Boston variables
multiple_linear_model = lm(medv~.-age, data = Boston) # perdictors are all except age
multiple_linear_model = update(multiple_linear_model, ~.-age) # instead of above

# Interaction terms
model = lm(medv ~ lstat*age) # --> lstat, age, lstat * age are predictors

# Non-linear Transformations
model = lm(medv ~ lstat + I(lstat^2), data = Boston)

plot(model)

# compare 2 models using hypothesis testing
model1 = lm(medv ~ lstat)
model2 = lm(medv ~ lstat + I(lstat^2))
anova(model1, model2) # performs hypothesis test --> H0 : two models fit the data equally well

# perform polynomial fit for a specific predictor
model = lm(medv ~ poly(lstat, 5))
