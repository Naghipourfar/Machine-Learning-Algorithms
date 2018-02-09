
install.packages("ISLR")
library(ISLR)
x = rnorm(50)
y = x + rnorm(50, mean = 50, sd = 0.1)
cor(x, y)
plot(x, y)

x = rnorm(50)
y = rnorm(50)
cor(x, y)
plot(x, y, xlab = "x-axis", ylab = "y-axis", main = "y(x)", col = "red")


y = x = seq(1, 10)
f = outer(x, y, function(x, y)cos(y)/(1 + x ^ 2))
contour(x, y, f)
contour(x, y, f, nlevels = 45, add = TRUE)
image(x, y, f, nlevels = 45, add = TRUE) # Produce a heatmap
fa = (f - t(f)) / 2
contour(x, y, fa, nlevels = 15)
image(x, y, fa, nlevels = 15)

persp(x, y, fa, theta = 0) # Produce a 3d-plot :D
persp(x, y, fa, theta = 30, phi = 50)


automobiles = Auto
View(automobiles) # Load some automobiles data
automobiles$cylinders = as.factor(automobiles$cylinders) # treat as a qualitative variable
plot(automobiles$cylinders, automobiles$mpg, col = "yellow")

pairs(automobiles) # Create scatter plot for all pair of variables in the given dataset






