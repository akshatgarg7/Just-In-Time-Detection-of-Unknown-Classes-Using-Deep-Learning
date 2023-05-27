### Contrastive Loss Function
How do we calculate the distance or the dissimilarity between these two vectors?

Let’s look at an example when both \(x^{(i)}\) and \(x^{(j)}\) are of the same person, I will denote these two images as \(A\) and \(B\). In this case, we want \(d\) to be a small value:

\(d(A, B) = ||f(A) – f(B)||^2\)

But, what happens when \(x^{(i)}\) and \(x^{(j)}\) are of different people, or a negative pair? Then the distance \(d\) should be large. In such a case, we will not apply the l2 distance norm, or distance function directly, but rather a hinge loss. Why is that?

We want to separate \(f(x^{(i)})\) and \(f(x^{(j)})\) if they are of different people, but we want to separate them until we hit a certain margin \(m\). The idea behind this, is that we don’t actually want to push \(f(x^{(i)})\) and \(f(x^{(j)})\) further and further apart if they are already far from the margin \(m\).

\(d(A, B) = max(0, m^2 – ||f(A) – f(B)||^2)\)

Now, we combine the two together and end up with a formula that is called the Contrastive Loss Function.

This function will calculate the similarity between the two vectors. As we mentioned earlier, the objective of the Siamese networks is not to classify, but rather to differentiate between the images. So a loss function like Cross-Entropy loss is not suitable for this problem.

The Contrastive Loss function evaluates how well the network distinguishes a given pair of images. The function is defined as follows:

\(L(A, B, Y) = (Y)*||f(A) – f(B)||^2 + (1-Y)*\{ max(0, m^2 – ||f(A) – f(B)||^2)\}\)

Looking at the equation at the top, \(Y\) here indicates the label. The label will be 1 if the two images are of the same person, and 0 if the images are of different persons.

The equation has two parts. One positive part, when the two images are of the same people, and one negative part, when the two images are of different people.

From left to right, the first part of the formula is the positive pair, we know when the two images are of the same person, our \(Y\) will be 1. We want to minimize the distance between these two embeddings when the two images are of the same person. So the first part of the equation will be used, but the second part will be ignored because \((1-Y)\) will equal to \(0\).

When we actually have a negative pair, \(Y\) is equal to 0, the left part is ignored and the second part is used. Here we apply the hinge loss, because of the reason mentioned above, and we separate our embeddings further and further away until we hit the margin \(M\).

The output of this equation will be the value that will indicate if the two images are of the same person or not, the same category or not.

This loss function is so to say, the most basic function for learning similarity. Still, even being the most basic, it can solve most of the similarity problems. 
