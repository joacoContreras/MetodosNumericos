
public class Main {
    public static void main(String[] args) {
        // Pruebas de la pila
        Pila<Integer> pila = new Pila<>();

        // Operaciones de prueba
        pila.push(10);
        pila.push(20);
        pila.push(30);
        
        System.out.println("Elemento en la cima (top): " + pila.top());
        pila.mostrarPila();

        System.out.println("Elemento eliminado (pop): " + pila.pop());
        pila.mostrarPila();

        System.out.println("Elemento eliminado (pop): " + pila.pop());
        System.out.println("Elemento en la cima después de pop: " + pila.top());
        pila.mostrarPila();
        
        System.out.println("La pila está vacía? " + pila.isEmpty());
    }
}
