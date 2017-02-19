/**
 * 
 */

package jmdn.method.classification.maxent;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;

import jmdn.base.struct.pair.PairIntDouble;

import javax.security.sasl.SaslServer;

import static java.lang.Math.abs;

/**
 * @author Xuan-Hieu Phan (pxhieu@gmail.com)
 * @version 1.1
 * @since 25-08-2014
 */
public class Inference {

    static int cntCorrect = 0;
    static int cntWrong = 0;

	protected Model model = null;

	protected int numLabels = 0;

    static double maxError = 0;

	// for classification
	protected double[] temp = null;

	/**
	 * Default class constructor.
	 */
	public Inference() {
		// do nothing
	}

	/**
	 * Initializing for doing inference.
	 */
	public void init() {
		numLabels = model.data.numLabels();
		temp = new double[numLabels];
	}

	/**
	 * Classifying for a given observation.
	 * 
	 * @param observation
	 *            the observation to be classified
	 */
	public synchronized void classify(Observation observation) {

        System.out.println(observation.originalData);

		int i;

		for (i = 0; i < numLabels; i++) {
			temp[i] = 0.0;
		}

		model.featureGenerator.startScanFeatures(observation);
		while (model.featureGenerator.hasNextFeature()) {
			Feature f = model.featureGenerator.nextFeature();

			temp[f.label] += model.lambda[f.index] * f.value;
		}

		double max = temp[0];
		int maxLabel = 0;
		for (i = 1; i < numLabels; i++) {
			if (max < temp[i]) {
				max = temp[i];
				maxLabel = i;
			}
		}

		observation.modelLabel = maxLabel;

        if (observation.isWrongPredict()) {
            double otherLabel = temp[0];
            if (Math.abs(otherLabel - max) < 0.000000001) otherLabel = temp[1];
            double value1 = Math.exp(max);
            double value0 = Math.exp(otherLabel);
            double normalize = (value0 + value1);
            double percent1 = value1/normalize;
            double percent0 = value0/normalize;
            System.out.println("-----------------------------");
            System.out.println(otherLabel + " " + value0 + " " + percent0 * 100);
            System.out.println(max + " " + value1 + " " + percent1 * 100);
            if (percent1*100 > 95.0 && observation.humanLabel == 0) {
                cntWrong += 1;
                System.out.print(observation.originalData);
                System.out.println(">>>>>>>>>>>>>>>>> " + cntWrong + " " + observation.humanLabel + " " + observation.modelLabel);
            }
        } else {
            double otherLabel = temp[0];
            if (Math.abs(otherLabel - max) < 0.000000001) otherLabel = temp[1];
            double value1 = Math.exp(max);
            double value0 = Math.exp(otherLabel);
            double normalize = (value0 + value1);
            double percent1 = value1/normalize;
            if (percent1*100 > 95.0 && observation.humanLabel == 0) {
                cntCorrect += 1;
                System.out.println(">>>>>>>>>>>>>>>>> " + cntCorrect + " " + cntWrong);
            }
        }
	}

	/**
	 * Getting the class distribution for a given observation.
	 * 
	 * @param observation
	 *            the input observation
	 * @return A probability distribution over classes of the input observation
	 */
	public synchronized List<PairIntDouble> getDistribution(Observation observation) {
		List<PairIntDouble> probs = new ArrayList<PairIntDouble>();

		int i;

		for (i = 0; i < numLabels; i++) {
			temp[i] = 0.0;
		}

		model.featureGenerator.startScanFeatures(observation);
		while (model.featureGenerator.hasNextFeature()) {
			Feature feature = model.featureGenerator.nextFeature();

			temp[feature.label] += model.lambda[feature.index] * feature.value;
		}

		for (i = 0; i < numLabels; i++) {
			probs.add(new PairIntDouble(i, temp[i]));
		}

		return probs;
	}

	/**
	 * Doing inference for a list of observations.
	 * 
	 * @param data
	 *            a list of input data observations
	 */
	public void doInference(List<Observation> data) {
		for (int i = 0; i < data.size(); i++) {
			Observation obsr = (Observation) data.get(i);

			classify(obsr);
		}
	}
}
