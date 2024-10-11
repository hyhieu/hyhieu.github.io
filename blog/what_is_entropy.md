---
layout: post
date: 2024-10-11
---

# What is entropy?

I have always had this question. No, the answers given in typical college probability classes do
not suffice. In fact, the shallowness of those answers disgust me. And yet, for many years, I have
been too lazy to read a rigorous theory about entropy. This changed after I found the excellent
[note by John Baez](). This post is my own reading note of this excellent writing.

## Entropy of an event
Let $X$ be an event that happens with probability $p \in [0, 1]$. We want a function that measures
*the amount of information we can tell if we observe $X$.* This function is just a map
$f: [0, 1] \to \mathbb{R}$, but we want it two important properties:

1. $f$ is strictly decreasing on $[0, 1]$. That is, if $0 \leq p_1 < p_2 \leq 1$,
   we want $f(p_1) > f(p_2)$. Why so? There is actually no *mathematical* reason. It's just that
   we the humans want to satisfy our own intuition. That intuition is:

   **When we observe an event that is *less likely* to happen, we learn more information.**

   That's it. To repeat myself, $f$ must be strictly decreasing on $[0, 1]$, just
   *because we want it to be so.*

2. For any $p_1, p_2 \in [0, 1]$, we have $f(p_1 p_2) = f(p_1) + f(p_2)$. Similar to the decreasing
   property, $f$ must be this way simply because we want it to be so (so that it describes our
   intuition about the world of random events).

   **If $X$ and $Y$ are indenpdent events that happen with probabilities $p_X$ and $p_Y$, then
   the information we learn when observing *both* $X$ and $Y$ is equal to the sum of information
   we learn when we observe $X$ and $Y$ individually.**

Here comes the interesting part. It is mathematically provable that any function $f$ that satisfies
both conditions above *must have the form*:

$$
\boxed{f(p) = a \cdot \ln{p}}
$$

for all $p \in [0, 1]$, where $a$ is a negative real constant.

<details>

<summary>
The proof is completely elementary. If you are curious, please read <a href="">John Baez's note</a>.
You can also try to prove it yourself.
</summary>

It's based on the theory of <a href="">Cauchy's functional equation</a>, which does not use any
knowledge that a high school student does not know. In fact, if a student practices high school
mathematical olympiad, they must have studied this proof technique.
</details>

## Nats and Bits
Thus far, we have established the desired properties of our *measurement of information,* which
then *force* such measurement to take a very specific form. However, we are still left with a
degree of freedom: the choise of the constant $a$. That is not good, because if two groups of
people choose two different constants, their measurements will be off by multiplicative constant.
<details markdown="1">
It's like like they are talking *pounds* and *kilograms*. By the way, I **hate** the imperial system!
</details>

To avoid such confusions, we define two units: Nats and Bits. For a choice of $a$, we define a
*bit* is $\boxed{-a \cdot \ln{2}}$, and a *nat* is $\boxed{-a}$.

With these definitions, a statement like *"this event has an entropy of 10 bits"* can be understood
as the event happens with probability $p$ where for *some* choice of $a < 0$, we have:

$$
\frac{f(p)}{-a \ln{2}} = 10 \times (-a \ln{2})
$$

The good thing is that this means:
$$
\frac{a \ln{p}}{-a \ln{2}} = -\log_2{p}= 10 \Longleftrightarrow p = 2^{-10} = 1/1024
$$

And the final interpretation is independent of the choice of $a$! So, when you hear the statement
*"this event has an entropy of 10 bits"*, you can operate with your favorite choice of $a$, as I
do with mine, but we understand that we both means $p = 1/1024$.

## Entropy of a distribution
What is a distribution, then? If you start telling me to *define* a probability distribution by
*imagining* that I am drawing a card from a deck, I will give you a middle finger.

A probability distribution is rigorously defined on a *measure space.* Let's go through some simple
definitions.

<details markdown="1">
<summary>Click for an adventure into measure theory.</summary>


### Measure theory

**Sample space.** Any set $\Omega$ can be a *sample space.*

**$\sigma$-algebra.** A $\sigma$-algebra on $\Omega$ is a collection $\mathcal{F}$ of subsets of $\Omega$
which has two properties:

  1. *Contains the empty set:* $\emptyset \in \mathcal{F}$.

  2. *Close under complement:* if $A \subseteq \Omega$ and $A \in \mathcal{F}$ then
     $\Omega - A \in \mathcal{F}$.

  3. *Close under countable union:* if $B \subseteq \mathbb{N}$ and $A_i \in \mathcal{F}$ for all
     $i \in B$ then $\bigcup_{i \in B} A_i \in \mathcal{F}$.

**Borel sets.** The Borel $\sigma$-algebra $B(\mathbb{R})$ is the smallest $\sigma$-algebra that
*contains all
open intervals in $\mathbb{R}$.

**Measure.** A measure on $(\Omega, \mathcal{F})$ is a map $\mu: \mathcal{F} \to [0, +\infty)$ that
has two properties:

  1. $\mu(\emptyset) = 0$.

  2. *Additive under countable addition:* if $B \subseteq \mathbb{N}$ and $A_i \in \mathcal{F}$
     for all $i \in B$ then: $\mu\big( \bigcup_{i \in B} A_i \big) = \sum_{i \in B} \mu(A_i)$.

**Probability measure.** A probability measure on $(\Omega, \mathcal{F})$ is a measure $P$ where
$P(\Omega) = 1$. Note that along with other axioms of a measure, this enforces $P: \mathcal{F} \to
[0, 1]$.

**Measureable function.** A function $f: \Omega \to \mathbb{R}$ is *measurable* with respect to
a $\sigma$-algebra $\mathcal{F}$ if for each Borel subset $B \in B(\mathbb{R})$, the pre-image of
$B$ via $f$ is in $\mathcal{F}$. That is:

$$
f^{-1}(B) := \{x \in \Omega : f(x) \in B \} \in \mathcal{F}
$$

</details>
