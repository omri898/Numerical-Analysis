A project I made for "Numerical Analysis" - a course in my second year at the University of Ben-Guriun.

Final grade: 94

## Assignment 1
- The function receives a function f, a range, and a number of points to use.
- The function returns another interpolated function g from samples of f.

## Assignment 2
- The function receives 2 functions- 𝑓1, 𝑓2, and a float maxerr.
- The function returns an iterable of approximate intersection Xs, such that:

    ∀𝑥 ∈ 𝑋, |𝑓1(𝑥) − 𝑓2(𝑥)| < 𝑚𝑎𝑥𝑒𝑟𝑟
    
    ∀𝑥𝑖𝑥𝑗 ∈ 𝑋, |𝑥𝑖 − 𝑥𝑗| > 𝑚𝑎𝑥𝑒𝑟𝑟

## Assignment 3
### Assignment3.integrate
- Receives a function f, a range, and a number of points n.
- Returns an approximation to the integral of the function f in the given range.

### Assignment3.areabetween
- Receives two functions 𝑓1, 𝑓2.
- Returns the area between 𝑓1, 𝑓2 .

## Assignment 4
- The function receives an input function that returns noisy results. The noise is normally distributed.
- Returns a function 𝑔 fitting the data sampled from the noisy function.

## Assignment 5
### Assignment5.area
- The function receives a shape contour.
- Returns the approximate area of the shape.

### Assignment5.fit_shape
- The function receives a generator that returns a point (x,y), that is close to the shape contour.
- Returns the area of the shape.
