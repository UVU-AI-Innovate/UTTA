#!/usr/bin/env python
"""
Generate a sample educational QA dataset for fine-tuning demonstrations.

This script creates a small but realistic dataset of educational questions and answers
that can be used to demonstrate different fine-tuning approaches.
"""

import os
import json
import argparse
from pathlib import Path
import random

# Categories of educational questions
CATEGORIES = [
    "mathematics",
    "science",
    "history",
    "literature",
    "computer_science"
]

# Sample questions and answers for each category
SAMPLE_QA = {
    "mathematics": [
        {
            "question": "Explain the concept of the Pythagorean theorem and provide an example of how it's used.",
            "answer": "The Pythagorean theorem states that in a right-angled triangle, the square of the length of the hypotenuse equals the sum of squares of the other two sides. Mathematically, if a and b are the legs of a right triangle and c is the hypotenuse, then a² + b² = c². For example, in a triangle with legs of 3 and 4 units, the hypotenuse would be 5 units, since 3² + 4² = 9 + 16 = 25, and √25 = 5. This theorem is used in various applications like construction, navigation, and calculating distances in coordinate geometry."
        },
        {
            "question": "What is calculus and why is it important in real-world applications?",
            "answer": "Calculus is a branch of mathematics that studies continuous change and motion through differential and integral calculus. Differential calculus breaks down complex problems into smaller, manageable parts through derivatives, while integral calculus combines these parts to find the whole through integration. It's crucial in real-world applications because it helps model and analyze dynamic systems. Engineers use it to design structures and circuits, physicists apply it to understand motion and forces, economists use it to optimize profit functions, medical researchers employ it to model disease spread, and computer scientists implement it in algorithms and machine learning. Essentially, calculus provides tools to understand and predict change in nearly every scientific and technical field."
        },
        {
            "question": "How do you solve a quadratic equation and what are the different methods available?",
            "answer": "A quadratic equation has the form ax² + bx + c = 0, where a ≠ 0. There are several methods to solve it:\n\n1. Factoring: If the equation can be written as (x-r)(x-s)=0, then x=r and x=s are the solutions.\n\n2. Quadratic Formula: x = (-b ± √(b² - 4ac)) / 2a. This works for any quadratic equation and gives both solutions.\n\n3. Completing the Square: Rearrange the equation to (x + b/2a)² = (b² - 4ac)/4a² and then solve for x.\n\n4. Graphical Method: Plot y = ax² + bx + c and find where the curve crosses the x-axis.\n\nThe choice of method depends on the specific equation. Factoring is quickest when possible, while the quadratic formula always works but requires more calculation."
        }
    ],
    "science": [
        {
            "question": "Explain the process of photosynthesis and why it's essential for life on Earth.",
            "answer": "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose (sugar). The process takes place primarily in the chloroplasts, specifically using the green pigment chlorophyll. During photosynthesis, plants take in carbon dioxide (CO₂) from the air and water (H₂O) from the soil, and in the presence of sunlight, convert them into glucose (C₆H₁₂O₆) and oxygen (O₂).\n\nThe chemical equation is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂\n\nPhotosynthesis is essential for life on Earth for several reasons: it produces oxygen that most organisms need for respiration; it converts inorganic carbon (CO₂) into organic compounds that form the base of food chains; it helps regulate atmospheric CO₂ levels, impacting climate; and all of our food comes either directly or indirectly from photosynthetic organisms."
        },
        {
            "question": "What are the fundamental forces in physics and how do they interact with matter?",
            "answer": "There are four fundamental forces in physics:\n\n1. Gravitational Force: The weakest force, but with infinite range. It attracts objects with mass to each other and is responsible for planetary orbits and our weight on Earth.\n\n2. Electromagnetic Force: Acts between electrically charged particles, having infinite range. It's responsible for electricity, magnetism, light, and holds atoms and molecules together.\n\n3. Strong Nuclear Force: The strongest force, but with very short range (within atomic nuclei). It binds protons and neutrons together in atomic nuclei, overcoming the electromagnetic repulsion between protons.\n\n4. Weak Nuclear Force: Acts on subatomic particles, with very short range. It's responsible for radioactive decay and plays a crucial role in nuclear fusion in stars.\n\nThese forces interact with matter through force carrier particles: gravitons (theoretical) for gravity, photons for electromagnetic force, gluons for strong force, and W and Z bosons for weak force. Modern physics aims to unify these forces into a single theory, with electromagnetism and the weak force already unified into the electroweak force."
        }
    ],
    "history": [
        {
            "question": "What were the main causes and consequences of World War I?",
            "answer": "World War I (1914-1918) was caused by several interconnected factors:\n\n1. Militarism: An arms race between European powers created tension and preparedness for war.\n2. Alliances: Europe was divided into two opposing alliance systems (Triple Alliance and Triple Entente).\n3. Imperialism: Competition for colonies and resources led to conflicts between powers.\n4. Nationalism: Strong patriotic sentiments increased hostility between nations.\n5. Immediate trigger: The assassination of Archduke Franz Ferdinand of Austria-Hungary in Sarajevo.\n\nThe consequences were far-reaching:\n\n1. Human cost: About 20 million deaths and 21 million wounded.\n2. Political changes: Collapse of four empires (German, Austro-Hungarian, Ottoman, and Russian).\n3. Treaty of Versailles: Harsh terms imposed on Germany, sowing seeds for WWII.\n4. New nations: Creation of new countries in Europe and the Middle East.\n5. Russian Revolution: Leading to the formation of the Soviet Union.\n6. Economic impact: Massive debts, economic instability, and the groundwork for the Great Depression.\n7. Social changes: Advances in women's rights, shifts in class structures.\n8. League of Nations: First international organization aimed at preventing future wars.\n\nThese consequences reshaped the global order and directly influenced events leading to World War II."
        },
        {
            "question": "Describe the significance of the Industrial Revolution and how it transformed society.",
            "answer": "The Industrial Revolution, occurring roughly from 1760 to 1840, marked humanity's transition from agrarian economies to industrialized ones. It began in Britain and spread to Western Europe and North America.\n\nKey transformations included:\n\n1. Economic changes: Shift from hand production to machine manufacturing, development of factory systems, growth of capitalism, increased productivity and wealth generation, and the rise of new economic theories.\n\n2. Technological advancements: Steam power, textile machinery, iron production techniques, improved transportation (railways, steamships), and communication systems.\n\n3. Social impact: Urbanization as people migrated to cities for factory work, emergence of a working class and middle class, harsh working conditions, child labor, widening wealth gaps, and new social movements advocating for workers' rights.\n\n4. Political consequences: Rise of labor movements, push for democratic reforms, new government regulations, and the emergence of socialism and communism as responses to industrial capitalism.\n\n5. Global effects: European industrial powers sought raw materials and markets abroad, intensifying colonialism and creating global economic interdependence.\n\nThe Industrial Revolution's significance lies in how it fundamentally altered not just how things were made, but the entire structure of society, work, family life, and global relationships. Its legacy continues today in ongoing debates about technology, labor rights, environmental impacts, and economic systems."
        }
    ],
    "literature": [
        {
            "question": "What are the major themes in Shakespeare's 'Hamlet' and why is this play still relevant today?",
            "answer": "Shakespeare's 'Hamlet' explores several major themes:\n\n1. Revenge: The central conflict revolves around Hamlet's struggle to avenge his father's murder.\n\n2. Mortality and the afterlife: The play contains deep philosophical explorations of death, exemplified in Hamlet's 'To be or not to be' soliloquy.\n\n3. Appearance vs. reality: Characters frequently hide behind masks of deception, while Hamlet seeks to uncover the truth.\n\n4. Madness: The line between Hamlet's feigned madness and actual mental deterioration remains ambiguous.\n\n5. Corruption and poison: Literal and metaphorical poison spreads through Denmark's court, symbolizing moral decay.\n\n6. Inaction vs. action: Hamlet's delay in carrying out his revenge raises questions about the relationship between thought and action.\n\nThe play remains relevant today because it examines timeless human concerns: moral dilemmas, the psychology of grief, political corruption, existential questions, family dynamics, and the complexity of human motivation. Its nuanced characterization and exploration of the human condition continue to resonate with contemporary audiences. The play's ambiguities also allow for endless reinterpretation, making it adaptable to different cultural contexts and time periods. Shakespeare's rich language and psychological insights ensure that 'Hamlet' continues to offer new perspectives on the human experience."
        },
        {
            "question": "Compare and contrast the literary movements of Romanticism and Realism.",
            "answer": "Romanticism (approximately 1800-1850) and Realism (approximately 1850-1900) represent two contrasting literary movements:\n\nRomanticism:\n- Emphasized emotion, imagination, and individualism\n- Celebrated nature as a spiritual force and source of inspiration\n- Featured exotic settings, supernatural elements, and the past (especially medieval)\n- Focused on the exceptional individual or hero\n- Prized originality and rejected conventional forms\n- Embraced idealism and sometimes escapism\n- Key authors: William Wordsworth, Samuel Taylor Coleridge, Victor Hugo, Edgar Allan Poe\n\nRealism:\n- Emphasized objective observation and factual representation\n- Portrayed everyday life and ordinary people\n- Featured contemporary settings and social issues\n- Focused on the typical rather than the exceptional\n- Aimed for accuracy and believability in representation\n- Embraced social criticism and reform\n- Key authors: Honoré de Balzac, Charles Dickens, George Eliot, Mark Twain\n\nThe transition between these movements reflected broader social changes: Romanticism emerged partly as a reaction against the Industrial Revolution and rationalism of the Enlightenment, while Realism developed as urbanization, science, and social problems demanded more direct engagement with contemporary reality.\n\nDespite their differences, both movements valued authenticity in different ways: Romantics sought authentic emotion and spiritual experience, while Realists sought authentic depiction of social conditions. Together, they demonstrate how literature both shapes and reflects the evolving concerns of society."
        }
    ],
    "computer_science": [
        {
            "question": "Explain the concept of object-oriented programming and its key principles.",
            "answer": "Object-Oriented Programming (OOP) is a programming paradigm that organizes code around data, or 'objects,' rather than functions and logic. An object is an instance of a class, which serves as a blueprint defining the object's structure.\n\nThe four key principles of OOP are:\n\n1. Encapsulation: Bundling data (attributes) and methods (functions) that operate on the data into a single unit (class), and restricting direct access to some of the object's components. This hides the internal state of objects and requires interaction through well-defined interfaces, reducing complexity and increasing security.\n\n2. Inheritance: The mechanism by which a class (child/derived class) can inherit attributes and methods from another class (parent/base class). This promotes code reuse and establishes a hierarchical relationship between classes.\n\n3. Polymorphism: The ability to present the same interface for different underlying data types. It allows methods to do different things based on the object they're acting upon, enabling a single action to be performed in different ways. This is often achieved through method overriding and method overloading.\n\n4. Abstraction: The concept of hiding complex implementation details and showing only the necessary features of an object. It helps manage complexity by allowing programmers to think about objects at a higher level.\n\nOOP is widely used because it helps manage complexity in large software systems, facilitates code reuse, makes code more modular and maintainable, and models real-world entities effectively."
        },
        {
            "question": "What is the difference between machine learning and deep learning?",
            "answer": "Machine learning (ML) and deep learning (DL) are related fields in artificial intelligence, with deep learning being a specialized subset of machine learning.\n\nMachine Learning:\n- Broader discipline where algorithms learn patterns from data without explicit programming\n- Can work effectively with smaller datasets\n- Often requires manual feature extraction and selection by human experts\n- Includes various approaches like linear regression, decision trees, SVMs, and random forests\n- Generally more interpretable, with clearer reasoning behind decisions\n- Less computationally intensive, can run on standard hardware\n- Better suited for structured data with clear features\n\nDeep Learning:\n- Subset of machine learning based on artificial neural networks with multiple layers\n- Typically requires large amounts of data to perform well\n- Performs automatic feature extraction, learning the relevant features directly from data\n- Primary architectures include CNNs (for images), RNNs and Transformers (for sequential data)\n- Often considered a \"black box\" with less interpretability\n- Computationally intensive, typically requires specialized hardware (GPUs/TPUs)\n- Excels with unstructured data like images, audio, and text\n\nThe key breakthrough of deep learning is its ability to learn hierarchical features automatically: early layers learn simple features, while deeper layers combine these into more complex concepts. This has enabled remarkable advances in computer vision, natural language processing, and reinforcement learning, though at the cost of increased computational requirements and reduced explainability compared to traditional machine learning approaches."
        }
    ]
}

def generate_dataset(num_examples=30, output_path="educational_qa_sample.jsonl"):
    """
    Generate a sample dataset of educational QA pairs.
    
    Args:
        num_examples: Number of examples to generate
        output_path: Path to save the generated dataset
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate examples with balanced categories
    examples_per_category = num_examples // len(CATEGORIES)
    remainder = num_examples % len(CATEGORIES)
    
    dataset = []
    for i, category in enumerate(CATEGORIES):
        # Add extra example to some categories if num_examples doesn't divide evenly
        category_examples = examples_per_category + (1 if i < remainder else 0)
        
        # Get samples for this category (with replacement if needed)
        category_samples = SAMPLE_QA[category]
        if category_examples > len(category_samples):
            # Duplicate samples if we need more than available
            category_samples = category_samples * (category_examples // len(category_samples) + 1)
        
        # Select random samples for this category
        selected_samples = random.sample(category_samples, category_examples)
        
        for sample in selected_samples:
            example = {
                "question": sample["question"],
                "answer": sample["answer"],
                "category": category
            }
            dataset.append(example)
    
    # Shuffle the dataset
    random.shuffle(dataset)
    
    # Write to JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example) + '\n')
    
    print(f"Generated {len(dataset)} examples and saved to {output_path}")
    print(f"Category distribution:")
    for category in CATEGORIES:
        count = sum(1 for ex in dataset if ex["category"] == category)
        print(f"  - {category}: {count} examples")
    
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a sample educational QA dataset")
    parser.add_argument(
        "--num_examples", 
        type=int, 
        default=30, 
        help="Number of examples to generate (default: 30)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="examples/data/educational_qa_sample.jsonl", 
        help="Path to save the dataset (default: examples/data/educational_qa_sample.jsonl)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Generate the dataset
    generate_dataset(args.num_examples, args.output_path) 