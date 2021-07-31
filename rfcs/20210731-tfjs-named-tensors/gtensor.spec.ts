// gtensor.spec.ts
import * as gtensor from './gtensor';
import { Dims } from './gtensor';
import * as tf from '@tensorflow/tfjs'

describe('gtensor', () => {
  beforeEach(() => {
  });

  it('creatingGTensors', () => {
    // Making a GTensor with an initializer:
    const g1 = gtensor.makeTruncNormal({ inputRep: 2, kqRep: 3 });
    // gshape() gives you the dict that describes the dimension's sizes.
    expect(g1.gshape()).toEqual({ inputRep: 2, kqRep: 3});

    // Making a GTensor from a tensor by naming the dimensions:
    const g2 = new gtensor.GTensor(tf.tensor(
          [ [ // 'example' dimension index 0
              [ // 'pos' dimension index 0: contains an array of repSize
                1, // 'repSize' dimension index 0
                2  // repSize index 1
              ],
              [3, 4], // pos index 1
              [5, 6], // pos index 2
            ],
            [ [1, 2], [3, 4], [5, 6], ], // example index 1
            [ [1, 2], [3, 4], [5, 6], ], // example index 2
            [ [1, 2], [3, 4], [5, 6], ], // example index 3
          ]
          ), ['example', 'pos', 'repSize']);
    expect(g2.gshape()).toEqual({example: 4, pos: 3, repSize: 2});
  });

  it('transpose', () => {
    const g1 = new gtensor.GTensor(tf.tensor(
      [ [ // example = 0
          [1, 2], // pos = 0
          [3, 4], // pos = 1
          [5, 6], // pos = 2
        ],
        [ // example = 1
          [1, 2], [3, 4], [5, 6],
        ],
        [ // example = 2
          [1, 2], [3, 4], [5, 6],
        ],
        [ // example = 3
          [1, 2], [3, 4], [5, 6],
        ],
      ]
      ), ['example', 'pos', 'repSize']);
    const x = typeof(g1)
    expect(g1.gshape).toEqual({'example': 3, 'pos': 3, 'repSize': 2});

    console.log('g1.gshape', g1.gshape());
    console.log(x);

    const g2 = g1.transpose();
    console.log(g1.dimNames);
    console.log(g1.tensor.shape);
    console.log(g2.dimNames);
    console.log(g2.tensor.shape);

    expect(g2.dimNames).toEqual(g1.dimNames.slice().reverse());
    expect(g2.tensor.shape).toEqual(g1.tensor.shape.slice().reverse());

    // GTensor is the thing that holds a dimension map with the tensor and dimNames.
    const bar = new gtensor.GTensor(
      tf.initializers.truncatedNormal({}).apply([1,2,3,4,5]),
      ['a', 'b', 'c', 'd', 'e', 'f']);

    // It's usually more convenient to work directly with dimension maps.
    const foo = gtensor.makeZeros({  x: 3, y: 2, c: 3 }).dim;

    // But they can work together easily too... for gtensors you just access
    // the dimension map via 'dim'
    gtensor.dot(foo.c, bar.dim.c);
    // gtensor.dot(foo.c, bar.dim.a)  // type error!
    // foo.c.dot(bar.dim.a);  // type error!
    foo.c.dot(bar.dim.c);
// ---------------
    const queryM = gtensor.makeTruncNormal({ inputRep: 2, kqRep: 3 }).dim;
    const keyM = gtensor.makeTruncNormal({ inputRep: 2, kqRep: 3 }).dim;
    const valueM = gtensor.makeTruncNormal({ inputRep: 2, valueRep: 4 }).dim;
    const oneInput = gtensor.makeTruncNormal({ seqLen: 8, inputRep: 2 }).dim;
    const batchedInput = gtensor.makeTruncNormal({ batchSize: 10, seqLen: 8, inputRep: 2 }).dim;

    // const inputKeys = inputM.dim.inputRep.dot(keyM.dim.inputRep).rename(
    //   'seqLen', 'seqLen2' )
    // const inputKeys = inputM.dim.inputRep.dot(keyM.dim.inputRep).renaming(
    //   { seqLen: 'keySeqLen'});

    function attentionHeadFn(input: Dims<'seqLen'|'inputRep'>): Dims<'seqLen'|'valueRep'> {
      const inputKeys = input.inputRep.dot(keyM.inputRep).seqLen.rename('keySeqLen');
      const inputQueries = input.inputRep.dot(queryM.inputRep);
      const attention = inputKeys.kqRep.dot(inputQueries.kqRep);
      const values = input.inputRep.dot(valueM.inputRep);
      const attendedValues = values.seqLen.dot(attention.seqLen).keySeqLen.rename('seqLen');
      return attendedValues;
    }

    // It's possible to make input be matched strictly, but you have to introduce `ExactDims`
    // wrapper and a new type parameter. :/
    interface Error_GivenHadExtraTypes<T> {
      _Error_GivenHadExtraTypes: ['Error_GivenHadExtraTypes', T];
    }

    type ExactDims<Exact extends string, Given extends string> =
      Exclude<Given,Exact> extends never ? Dims<Given> : Error_GivenHadExtraTypes<Exclude<Given,Exact>>;

    function attentionHeadFn2<T extends string>(
        maybeInput: ExactDims<'seqLen'|'inputRep',T>): Dims<'seqLen'|'valueRep'> {
      const input = maybeInput as never as Dims<'seqLen'|'inputRep'>;
      const inputKeys = input.inputRep.dot(keyM.inputRep).seqLen.rename('keySeqLen');;
      const inputQueries = input.inputRep.dot(queryM.inputRep);
      const attention = inputKeys.kqRep.dot(inputQueries.kqRep);
      const values = input.inputRep.dot(valueM.inputRep);
      const attendedValues = values.seqLen.dot(attention.seqLen).keySeqLen.rename('seqLen');
      return attendedValues;
    }
    // Bug/TODO: extra dimensions don't get caught by type-checker. :(
    //   const attendedValues = attentionHeadFn(batchedInput);
    // Maybe we have to use record types instead of simple string unions...
    const attendedValues = attentionHeadFn(oneInput);

    //const attendedValues2 = attentionHeadFn2(batchedInput); // Has error, yay, but what a mess...
    const attendedValues3 = attentionHeadFn2(oneInput);

    const batchedAttentionHeadFn = gtensor.liftFnOverDim('batchSize', attentionHeadFn);
    const batchedAttendedValues = batchedAttentionHeadFn(batchedInput);

    //
    function attentionHeadFn3(input: Dims<'seqLen'|'inputRep'>):
    { attendedValues: Dims<'seqLen'|'valueRep'>,
      inputKeys: Dims<"kqRep" | "keySeqLen">,
      inputQueries: Dims<"kqRep" | "seqLen">,
      attention: Dims<"seqLen" | "keySeqLen">,
      values: Dims<"valueRep" | "seqLen">,
    } {
      const inputKeys = input.inputRep.dot(keyM.inputRep).seqLen.rename('keySeqLen');
      const inputQueries = input.inputRep.dot(queryM.inputRep);
      const attention = inputKeys.kqRep.dot(inputQueries.kqRep);
      const values = input.inputRep.dot(valueM.inputRep);
      const attendedValues = values.seqLen.dot(attention.seqLen).keySeqLen.rename('seqLen');
      return { attendedValues, inputKeys, inputQueries, attention, values };
    }
    // TODO: this should error, but does not, see ExactDims trick above.
    // attentionHeadFn3(batchedInput);

    const batchedAttentionHeadFn3 = gtensor.liftMapFnOverDim('batchSize', attentionHeadFn3);
    const outputs = batchedAttentionHeadFn3(batchedInput);
  })
});

