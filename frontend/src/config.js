/**
 * App-wide dataset configuration.
 *
 * The dataset target is Moldovan-accented speech ONLY — content tagged with
 * other regions (RO, DIASPORA, UNKNOWN) is collected by the pipeline but
 * counts as "outside target" in the Stats views.
 */

/** The region code that the dataset is built around. */
export const TARGET_REGION = 'MD';

/** Target size of the MD dataset, in hours of extracted speech. */
export const TARGET_HOURS = 200;
