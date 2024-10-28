import java.util.ArrayList;
import java.util.List;

    public class Pila<T> {
    // Lista para almacenar los elementos de la pila
    private List<T> elementos;

    // Constructor
    public Pila() {
        elementos = new ArrayList<>();
    }

    // Método push para agregar un elemento en la cima de la pila
    public void push(T elemento) {
        elementos.add(elemento);
    }

    // Método pop para eliminar el elemento en la cima de la pila
    public T pop() {
        if (!elementos.isEmpty()) {
            return elementos.remove(elementos.size() - 1);
        } else {
            System.out.println("La pila está vacía.");
            return null;
        }
    }

    // Método top para ver el elemento en la cima sin eliminarlo
    public T top() {
        if (!elementos.isEmpty()) {
            return elementos.get(elementos.size() - 1);
        } else {
            System.out.println("La pila está vacía.");
            return null;
        }
    }

    // Método para verificar si la pila está vacía
    public boolean isEmpty() {
        return elementos.isEmpty();
    }

    // Método para imprimir los elementos de la pila
    public void mostrarPila() {
        System.out.println("Elementos en la pila: " + elementos);
    }

}