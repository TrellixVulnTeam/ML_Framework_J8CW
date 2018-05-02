<h1>Spongebob Character Classifier</h1>
<h3>Overview</h3>
<p>This machine learning system will classify your favorite Spongebob Squarepants characters&mdash;assuming your favorite characters
are among the following:</p>
<ul>
    <li>Spongebob Squarepants</li>
    <li>Plankton</li>
    <li>Gary</li>
    <li>Patrick Star</li>
    <li>Sandy Cheeks</li>
    <li>Mr. Krabs</li>
    <li>Squidward</li>
</ul>

<h3>Details</h3>

<p>This project implements a three layer standard neural network. The overall architecture is as follows:</p>
<ul>
    <li>Input Layer --> 98 flattened image examples, each of original shape 100 x 100 x 3 (RGB)</li>
    <li>Hidden Layer --> A fully connected layer of input units 30000 and output units 10</li>
    <li>Activation --> ReLU activation layer</li>
    <li>Output Layer --> A fully connected layer of input units 10 and output units 7 (the number of classes)</li>
    <li>Output Activation --> Softmax activation layer</li>
</ul>

<p>The F1 scores for the datasets are:</p>
<ul>
    <li>Train: 100%
    <li>Cross Validation: 96%
    <li>Test: 98%</li>
</ul>

<h3>Instructions</h3>
<p>To test the tool:</p>
<ol>
    <li>Download or clone this repository
    <li>Find a quality image of your favorite Spongebob Character</li>
    <li>Save the image to the directory: <em>SpongebobCharacterClassifier/datasets/user_images</em> (make sure there is only 1 image in the directory)</li>
    <li>Run <em>main.py</em></li>
</ol>

<p>Once the process finishes, you should see a plot of your chosen character image, labeled with the correct character name :)</p>

<h3>Future Plans</h3>
The system does a poor job on Mr. Krabs, prefering to predict Patrick Star instead. Aside from their similar shape, I'm not immediately sure why this could be.
Next steps would be to address this shortcoming.</p>
