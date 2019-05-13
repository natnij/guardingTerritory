package territoryGame;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Combination<T> {
    private final List<List<T>> elements;

    public Combination(List<List<T>> elements) {
        this.elements = Collections.unmodifiableList(elements);
    }

    public List<T> get(int index) {
        List<T> result = new ArrayList<>();
        for(int i = elements.size() - 1; i >= 0; i--) {
            List<T> counter = elements.get(i);
            int counterSize = counter.size();
            result.add(counter.get(index % counterSize));
            index /= counterSize;
        }
        return result;
    }

    public int size() {
        int result = 1;
        for(List<T> next: elements) result *= next.size();
        return result;
    }
}